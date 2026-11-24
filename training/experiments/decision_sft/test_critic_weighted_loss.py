#!/usr/bin/env python3
"""CPU unit tests for the experiment-local critic loss implementation."""

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("critic_weighted_entry.py")
SPEC = importlib.util.spec_from_file_location("critic_weighted_entry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def close(actual, expected, atol=1e-6):
    if not torch.isclose(torch.as_tensor(actual), torch.as_tensor(expected), atol=atol):
        raise AssertionError(f"actual={actual} expected={expected}")


def test_unpacked_per_sample_and_class_balance():
    losses = torch.tensor([
        [1.0, 3.0, 100.0, 100.0],
        [5.0, 1.0, 9.0, 100.0],
    ])
    labels = torch.tensor([
        [1, 2, -100, -100],
        [1, 2, 3, -100],
    ])
    # Clean is encoded by a positive sentinel factor; error uses raw weights.
    scales = torch.tensor([
        [10000.0, 1000.0, 0.0, 0.0],
        [10.0, 0.5, 0.25, 0.0],
    ])
    objective, clean, error = MODULE.balanced_weighted_loss(
        losses, labels, scales, [(0, 0, 4), (1, 0, 4)]
    )
    expected_clean = (1.0 * 10.0 + 3.0 * 1.0) / 11.0
    expected_error = (5.0 * 10.0 + 1.0 * 0.5 + 9.0 * 0.25) / 10.75
    close(clean[0], expected_clean)
    close(error[0], expected_error)
    close(objective, 0.5 * expected_clean + 0.5 * expected_error)


def test_packed_boundaries_are_sample_normalized():
    losses = torch.tensor([[1.0, 3.0, 5.0, 1.0, 9.0]])
    labels = torch.tensor([[1, 2, 1, 2, 3]])
    scales = torch.tensor([[10000.0, 1000.0, 10.0, 0.5, 0.25]])
    objective, clean, error = MODULE.balanced_weighted_loss(
        losses, labels, scales, [(0, 0, 2), (0, 2, 5)]
    )
    close(objective, 0.5 * clean[0] + 0.5 * error[0])


def test_description_split_roundtrip():
    text = ',"boxes":[{"bbox_1000_xyxy":[1,2,3,4],"family":"x","description":"visible hand defect"}]}'
    parts, weights = MODULE._split_description_segments(text)
    assert "".join(parts) == text
    description_index = parts.index("visible hand defect")
    close(weights[description_index], MODULE.DESCRIPTION_WEIGHT)


def test_optional_empty_think_prefix_is_zero_weighted():
    MODULE._register_loss_scale()
    from swift.loss_scale import loss_scale_map
    scaler = loss_scale_map["critic_decision_v1"]()
    parts, weights = scaler.get_loss_scale('<think>\n\n</think>\n\n{"no_error":true,"boxes":[]}')
    assert "".join(parts).endswith('{"no_error":true,"boxes":[]}')
    close(weights[0], 0.0)
    close(weights[1], MODULE.DECISION_WEIGHT)
    suffix_parts, suffix_weights = scaler.get_loss_scale('<|im_end|>\n')
    assert suffix_parts == ['<|im_end|>\n']
    close(suffix_weights[0], 1.0)


if __name__ == "__main__":
    test_unpacked_per_sample_and_class_balance()
    test_packed_boundaries_are_sample_normalized()
    test_description_split_roundtrip()
    test_optional_empty_think_prefix_is_zero_weighted()
    print("critic weighted loss unit tests: PASS")
