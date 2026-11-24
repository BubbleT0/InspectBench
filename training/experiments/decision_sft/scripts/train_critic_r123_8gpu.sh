#!/bin/bash
set -euo pipefail

VARIANT=${1:-}
case "${VARIANT}" in
  r1)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_neutral_defaultloss_lr1e6
    ;;
  r2)
    LOSS_MODE=critic_decision_v1
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r2_neutral_weightedloss_lr1e6
    ;;
  r3)
    LOSS_MODE=critic_decision_v1
    LR=5e-7
    MIN_LR=5e-8
    RUN_NAME=r3_neutral_weightedloss_lr5e7
    ;;
  *)
    echo "usage: $0 {r1|r2|r3}" >&2
    exit 2
    ;;
esac

EXP_DIR=/path/to/experiments/decision_sft
SWIFT_HOME=/path/to/storage/multimodal_ckpt/research/autotrain/ms-swift
ENV_PATH=/path/to/storage/research/miniconda3/envs/swift
AOSS_CONF=/path/to/storage/research/aoss.conf
MODEL_PATH=/path/to/checkpoints/workflow_sft
DATA_DIR=${DATA_DIR_OVERRIDE:-${EXP_DIR}/data/dev_4000_800}
TRAIN_JSONL=${DATA_DIR}/train.jsonl
VAL_JSONL=${DATA_DIR}/validation.jsonl
ENTRY=${EXP_DIR}/scripts/critic_weighted_entry.py
CACHE_DIR=${CACHE_DIR_OVERRIDE:-${EXP_DIR}/cache_full_common_v1}
OUTPUT_DIR=${OUTPUT_DIR_OVERRIDE:-/path/to/storage/multimodal_ckpt/research/autotrain/output/critic_singlemodel_r123_8gpu_20260824/${RUN_NAME}}
LOG_DIR=${LOG_DIR_OVERRIDE:-${EXP_DIR}/logs/${RUN_NAME}}

TP=2
PP=4
CP=1
VPP=2
NPROC_PER_NODE=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
MAX_LENGTH=8192
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS_OVERRIDE:-1}
MAX_STEPS=${MAX_STEPS_OVERRIDE:-}
SAVE_STEPS=${SAVE_STEPS_OVERRIDE:-50}
EVAL_STEPS=${EVAL_STEPS_OVERRIDE:-50}
DATASET_NUM_PROC=${DATASET_NUM_PROC_OVERRIDE:-16}

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
NODE_RANK=${RANK:-${OMPI_COMM_WORLD_RANK:-0}}
NNODES=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
MASTER_PORT=${MASTER_PORT:-29541}
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/RANK_${NODE_RANK}_${RUN_STAMP}.log
exec > >(tee "${LOG_FILE}") 2>&1

echo "variant=${VARIANT} run_name=${RUN_NAME}"
echo "experiment=${EXP_DIR}"
echo "model=${MODEL_PATH}"
echo "loss_mode=${LOSS_MODE} lr=${LR} min_lr=${MIN_LR}"
echo "train=${TRAIN_JSONL} validation=${VAL_JSONL}"
echo "cache=${CACHE_DIR} output=${OUTPUT_DIR}"
echo "topology=TP${TP}_PP${PP}_CP${CP}_VPP${VPP} nodes=${NNODES} total_gpus=$((NNODES * NPROC_PER_NODE))"
echo "packing=false dataset_shuffle=false repeat_time=1.0"

if [ "${NNODES}" -ne 1 ]; then
  echo "expected one 8-GPU node, got NNODES=${NNODES}" >&2
  exit 2
fi
for required in "${SWIFT_HOME}" "${ENV_PATH}" "${MODEL_PATH}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${ENTRY}"; do
  if [ ! -e "${required}" ]; then
    echo "required path missing: ${required}" >&2
    exit 2
  fi
done

source "${ENV_PATH%/envs/*}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${SWIFT_HOME}"

export AOSS_CONF
export PYTHONNOUSERSITE=1
export PYTHONPATH=${SWIFT_HOME}/dependencies/Megatron-LM:${SWIFT_HOME}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=${EXP_DIR}/cache/torchinductor
export TRITON_CACHE_DIR=${EXP_DIR}/cache/triton
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=1
export USE_PYTORCH_MEM_PROFILER=0
export IMAGE_MAX_TOKEN_NUM=2048
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export MAX_PIXELS=200704
export SWIFT_USE_MCORE_GDN=1

export_cache() {
  local split=$1
  local jsonl=$2
  local out=${CACHE_DIR}/${split}
  if [ -d "${out}/train" ]; then
    echo "cache_skip split=${split} path=${out}/train"
    return
  fi
  echo "cache_start split=${split} loss_mode=${LOSS_MODE}"
  if [ "${LOSS_MODE}" = "default" ]; then
    python3 -u swift/cli/export.py \
      --model "${MODEL_PATH}" \
      --dataset "${jsonl}" \
      --split_dataset_ratio 0 \
      --dataset_num_proc "${DATASET_NUM_PROC}" \
      --dataset_shuffle false \
      --data_seed 20260824 \
      --to_cached_dataset true \
      --packing false \
      --enable_thinking false \
      --add_non_thinking_prefix false \
      --response_prefix "" \
      --loss_scale default \
      --truncation_strategy delete \
      --max_length "${MAX_LENGTH}" \
      --output_dir "${out}"
  else
    python3 -u "${ENTRY}" export \
      --model "${MODEL_PATH}" \
      --dataset "${jsonl}" \
      --split_dataset_ratio 0 \
      --dataset_num_proc "${DATASET_NUM_PROC}" \
      --dataset_shuffle false \
      --data_seed 20260824 \
      --to_cached_dataset true \
      --packing false \
      --enable_thinking false \
      --add_non_thinking_prefix false \
      --response_prefix "" \
      --loss_scale "${LOSS_MODE}" \
      --truncation_strategy delete \
      --max_length "${MAX_LENGTH}" \
      --output_dir "${out}"
  fi
  test -d "${out}/train"
  echo "cache_complete split=${split} path=${out}/train"
}

export_cache train "${TRAIN_JSONL}"
export_cache validation "${VAL_JSONL}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "preflight_complete variant=${VARIANT} cache=${CACHE_DIR}"
  exit 0
fi

COMMON_ARGS=(
  --model "${MODEL_PATH}"
  --save_safetensors true
  --load_from_cache_file true
  --loss_scale "${LOSS_MODE}"
  --enable_thinking false
  --add_non_thinking_prefix false
  --response_prefix ""
  --cached_dataset "${CACHE_DIR}/train/train"
  --cached_val_dataset "${CACHE_DIR}/validation/train"
  --pipeline_model_parallel_size "${PP}"
  --tensor_model_parallel_size "${TP}"
  --context_parallel_size "${CP}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --packing false
  --padding_free false
  --dataset_shuffle false
  --val_dataset_shuffle false
  --seed 20260824
  --data_seed 20260824
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --virtual_pipeline_model_parallel_size "${VPP}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --finetune true
  --cross_entropy_loss_fusion true
  --cross_entropy_fusion_impl te
  --packing_cp_align false
  --lr "${LR}"
  --lr_warmup_fraction 0.2
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps 1
  --save_steps "${SAVE_STEPS}"
  --max_length "${MAX_LENGTH}"
  --dataloader_num_workers 1
  --no_save_optim true
  --no_save_rng true
  --dataset_num_proc 1
  --sequence_parallel true
  --attention_backend flash
  --strict false
  --truncation_strategy delete
  --vit_gradient_checkpointing false
  --freeze_llm false
  --freeze_vit true
  --overlap-grad-reduce
  --overlap-param-gather
  --report_to tensorboard
  --profile false
  --use_pytorch_profiler false
)

if [ -n "${MAX_STEPS}" ]; then
  COMMON_ARGS+=(--train_iters "${MAX_STEPS}")
fi

if [ "${ARGS_PREFLIGHT_ONLY:-0}" = "1" ]; then
  python3 -u "${ENTRY}" parse_train "${COMMON_ARGS[@]}"
  exit 0
fi

if [ "${LOSS_MODE}" = "default" ]; then
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    swift/cli/_megatron/sft.py "${COMMON_ARGS[@]}"
else
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${ENTRY}" train "${COMMON_ARGS[@]}"
fi

echo "training_complete variant=${VARIANT} output=${OUTPUT_DIR}"
