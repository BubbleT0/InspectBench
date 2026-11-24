#!/usr/bin/env python3
"""Register critic decision weighting and run ms-swift export or Megatron SFT.

This is an experiment-local wrapper. It does not modify the shared ms-swift tree.
Positive loss-scale sentinels encode the ground-truth class inside the training
batch only. Clean scales are multiplied by CLEAN_CLASS_SENTINEL and decoded back
before loss computation. The model never receives this metadata.
"""

from __future__ import annotations

import re
import sys
from typing import Any


DECISION_WEIGHT = 10.0
CLEAN_REMAINDER_WEIGHT = 1.0
ERROR_LOCALIZATION_WEIGHT = 0.5
DESCRIPTION_WEIGHT = 0.25
CLEAN_CLASS_SENTINEL = 1000.0


def _split_description_segments(text: str) -> tuple[list[str], list[float]]:
    pattern = re.compile(r'("description":")([^"]*)(")')
    parts: list[str] = []
    weights: list[float] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            parts.append(text[cursor:match.start()])
            weights.append(ERROR_LOCALIZATION_WEIGHT)
        parts.extend([match.group(1), match.group(2), match.group(3)])
        weights.extend([ERROR_LOCALIZATION_WEIGHT, DESCRIPTION_WEIGHT, ERROR_LOCALIZATION_WEIGHT])
        cursor = match.end()
    if cursor < len(text):
        parts.append(text[cursor:])
        weights.append(ERROR_LOCALIZATION_WEIGHT)
    return parts, weights


def _register_loss_scale() -> None:
    from swift.loss_scale import LossScale, loss_scale_map

    class CriticDecisionLossScale(LossScale):
        is_binary = False

        def get_loss_scale(self, context: str, **kwargs):
            if not isinstance(context, str):
                return super().get_loss_scale(context, **kwargs)
            match = re.search(r'(\{"no_error":(?:true|false))', context)
            if match is None:
                # Assistant suffixes such as <|im_end|> are passed separately.
                # Their sign is applied by __call__ after the target is validated.
                return [context], [1.0]
            prefix = context[:match.start()]
            decision = match.group(1)
            remainder = context[match.end():]
            is_clean = decision.endswith("true")
            if is_clean:
                return [prefix, decision, remainder], [0.0, DECISION_WEIGHT, CLEAN_REMAINDER_WEIGHT]
            rest_parts, rest_weights = _split_description_segments(remainder)
            return [prefix, decision] + rest_parts, [0.0, DECISION_WEIGHT] + rest_weights

        def __call__(self, context_list, context_types, messages, **kwargs):
            contexts, weights = super().__call__(context_list, context_types, messages, **kwargs)
            assistant_targets = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
            if not assistant_targets:
                return contexts, weights
            target = assistant_targets[-1]
            if target.startswith('{"no_error":true'):
                factor = CLEAN_CLASS_SENTINEL
            elif target.startswith('{"no_error":false'):
                factor = 1.0
            else:
                raise ValueError(f"unable to infer critic class from target: {target[:160]!r}")
            # ms-swift masks every token whose loss_scale is <= 0, so a negative
            # class sign would silently drop all clean targets. Keep every
            # supervised scale positive and decode the sentinel in loss_func.
            return contexts, [0.0 if float(w) == 0.0 else factor * abs(float(w)) for w in weights]

    loss_scale_map["critic_decision_v1"] = CriticDecisionLossScale


def balanced_weighted_loss(
    losses,
    labels,
    signed_scales,
    boundaries: list[tuple[int, int, int]],
):
    """Return class-balanced mean of per-sample weighted token means.

    boundaries contains (batch_row, start, end) for each original sample.
    """
    import torch

    clean_losses = []
    error_losses = []
    for batch_row, start, end in boundaries:
        token_loss = losses[batch_row, start:end]
        token_labels = labels[batch_row, start:end]
        token_scales = signed_scales[batch_row, start:end]
        valid = (token_labels != -100) & (token_scales != 0)
        if not torch.any(valid):
            continue
        scales = token_scales[valid]
        is_clean = bool(torch.max(scales).item() >= CLEAN_CLASS_SENTINEL)
        weights = scales / CLEAN_CLASS_SENTINEL if is_clean else scales
        sample_loss = (token_loss[valid] * weights).sum() / weights.sum().clamp_min(1e-12)
        if is_clean:
            clean_losses.append(sample_loss)
        else:
            error_losses.append(sample_loss)

    if not clean_losses and not error_losses:
        return losses.sum() * 0.0, clean_losses, error_losses
    if clean_losses and error_losses:
        objective = 0.5 * torch.stack(clean_losses).mean() + 0.5 * torch.stack(error_losses).mean()
    else:
        objective = torch.stack(clean_losses or error_losses).mean()
    return objective, clean_losses, error_losses


def _register_trainer_patch() -> None:
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from swift.megatron.trainers.trainer import MegatronTrainer

    original_loss_func = MegatronTrainer.loss_func

    def critic_loss_func(
        self,
        output_tensor: torch.Tensor,
        *,
        labels: torch.Tensor,
        loss_scale: torch.Tensor | None = None,
        channels=None,
        packed_seq_params=None,
    ):
        if loss_scale is None:
            return original_loss_func(
                self,
                output_tensor,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params,
            )
        args = self.args
        if args.context_parallel_size != 1:
            raise ValueError("critic_decision_v1 currently requires context_parallel_size=1")

        losses = output_tensor.float()
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())

        if packed_seq_params is None:
            boundaries = [(row, 0, labels.shape[1]) for row in range(labels.shape[0])]
            num_samples = labels.shape[0]
            content_tokens = labels.numel()
        else:
            num_samples = int(packed_seq_params.num_samples)
            cu = packed_seq_params.cu_seqlens_q[:num_samples + 1]
            boundaries = [(0, int(cu[i].item()), int(cu[i + 1].item())) for i in range(num_samples)]
            content_tokens = int(cu[-1].item())

        objective, clean_losses, error_losses = balanced_weighted_loss(
            losses, labels, loss_scale, boundaries
        )
        reporting = torch.tensor(
            [float(objective.detach()), 1.0], dtype=torch.float32, device=labels.device
        )
        if dist.is_initialized():
            dist.all_reduce(reporting, group=mpu.get_data_parallel_group())

        def metric_pair(values):
            if values:
                return torch.stack([torch.stack(values).detach().sum(), torch.tensor(float(len(values)), device=labels.device)])
            return torch.tensor([0.0, 0.0], dtype=torch.float32, device=labels.device)

        total_tokens = labels.numel()
        metrics: dict[str, Any] = {
            "loss": reporting,
            "critic_clean_sample_loss": metric_pair(clean_losses),
            "critic_error_sample_loss": metric_pair(error_losses),
            "critic_clean_samples": torch.tensor([len(clean_losses), 1], dtype=torch.float32, device=labels.device),
            "critic_error_samples": torch.tensor([len(error_losses), 1], dtype=torch.float32, device=labels.device),
            "num_samples": torch.tensor([num_samples, 1], dtype=torch.float32, device=labels.device),
            "content_tokens": torch.tensor([content_tokens, 1], dtype=torch.float32, device=labels.device),
            "padding_tokens": torch.tensor([total_tokens - content_tokens, 1], dtype=torch.float32, device=labels.device),
            "total_tokens": torch.tensor([total_tokens, 1], dtype=torch.float32, device=labels.device),
        }
        # Keep diagnostic metrics rank-local in the loss callback.  Issuing extra
        # DP collectives here can interleave with Megatron pipeline P2P/gradient
        # collectives and deadlock a multi-node DP=2 run.  The single reporting
        # loss reduction above matches the upstream ms-swift execution path.

        # Megatron divides the numerator by this value and then by num_microbatches.
        # The objective is already a normalized mean, so the correct divisor is one.
        local_normalizer = torch.tensor(1, dtype=torch.int, device=labels.device)
        return objective, local_normalizer, metrics

    MegatronTrainer.loss_func = critic_loss_func


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "parse_train", "train"}:
        raise SystemExit("usage: critic_weighted_entry.py {export|parse_train|train} [ms-swift args...]")
    mode = sys.argv.pop(1)
    _register_loss_scale()
    if mode == "export":
        from swift.pipelines import export_main
        export_main()
    elif mode == "parse_train":
        from swift.megatron.arguments import MegatronSftArguments
        from swift.utils import parse_args

        args, remaining = parse_args(MegatronSftArguments, sys.argv[1:])
        if remaining:
            raise ValueError(f"unparsed training arguments: {remaining}")
        print(
            "training argument parse: PASS "
            f"loss_scale={args.loss_scale} train_iters={args.train_iters} "
            f"tp={args.tensor_model_parallel_size} pp={args.pipeline_model_parallel_size} "
            f"cp={args.context_parallel_size} gbs={args.global_batch_size}"
        )
    else:
        _register_trainer_patch()
        from swift.megatron import megatron_sft_main
        megatron_sft_main()


if __name__ == "__main__":
    main()
