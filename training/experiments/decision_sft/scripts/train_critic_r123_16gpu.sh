#!/bin/bash
set -euo pipefail

VARIANT=${1:-}
TRAIN_ITERS=
NUM_TRAIN_EPOCHS=1
SAVE_STEPS=50
EVAL_STEPS=50
OVERLAP_P2P_COMM=true
VARIABLE_SEQ_LENGTHS=true
PACKING=false
PADDING_FREE=false
USE_VAL=true
CACHE_NAME=cache_full_common_v1
case "${VARIANT}" in
  r1)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_neutral_defaultloss_lr1e6_16gpu_dp2
    ;;
  r1_formal_packed)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_formal_packed8192_epoch1_lr1e6_16gpu_dp2
    PACKING=true
    PADDING_FREE=true
    USE_VAL=true
    CACHE_NAME=cache_full_common_v1
    ;;
  r1_np2p)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_diag_no_p2p_overlap_2step_16gpu_dp2
    TRAIN_ITERS=2
    NUM_TRAIN_EPOCHS=
    SAVE_STEPS=2
    EVAL_STEPS=2000000000
    OVERLAP_P2P_COMM=false
    ;;
  r1_fixed)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_diag_fixed8192_no_p2p_overlap_2step_16gpu_dp2
    TRAIN_ITERS=2
    NUM_TRAIN_EPOCHS=
    SAVE_STEPS=2
    EVAL_STEPS=2000000000
    OVERLAP_P2P_COMM=false
    VARIABLE_SEQ_LENGTHS=false
    ;;
  r1_packed)
    LOSS_MODE=default
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r1_diag_packed8192_2step_16gpu_dp2
    TRAIN_ITERS=2
    NUM_TRAIN_EPOCHS=
    SAVE_STEPS=2
    EVAL_STEPS=2000000000
    PACKING=true
    PADDING_FREE=true
    USE_VAL=false
    CACHE_NAME=cache_full_common_v1
    ;;
  r2_packed)
    LOSS_MODE=critic_decision_v1
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r2_diag_packed8192_2step_16gpu_dp2
    TRAIN_ITERS=2
    NUM_TRAIN_EPOCHS=
    SAVE_STEPS=2
    EVAL_STEPS=2000000000
    PACKING=true
    PADDING_FREE=true
    USE_VAL=false
    CACHE_NAME=cache_full_common_v1
    ;;
  r2_formal_packed)
    LOSS_MODE=critic_decision_v1
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r2_formal_packed8192_epoch1_lr1e6_16gpu_dp2
    PACKING=true
    PADDING_FREE=true
    USE_VAL=true
    CACHE_NAME=cache_full_common_v1
    ;;
  r2)
    LOSS_MODE=critic_decision_v1
    LR=1e-6
    MIN_LR=1e-7
    RUN_NAME=r2_neutral_weightedloss_lr1e6_16gpu_dp2
    ;;
  r3)
    LOSS_MODE=critic_decision_v1
    LR=5e-7
    MIN_LR=5e-8
    RUN_NAME=r3_neutral_weightedloss_lr5e7_16gpu_dp2
    ;;
  r3_formal_packed)
    LOSS_MODE=critic_decision_v1
    LR=5e-7
    MIN_LR=5e-8
    RUN_NAME=r3_formal_packed8192_epoch1_lr5e7_16gpu_dp2
    PACKING=true
    PADDING_FREE=true
    USE_VAL=true
    CACHE_NAME=cache_full_common_v1
    ;;
  *)
    echo "usage: $0 {r1|r1_formal_packed|r1_np2p|r1_fixed|r1_packed|r2_packed|r2_formal_packed|r2|r3|r3_formal_packed}" >&2
    exit 2
    ;;
esac

EXP_DIR=/path/to/experiments/decision_sft
FROZEN_STACK=${EXP_DIR}/deps/frozen_b8241e8_mg31e327_mb_c527134
SWIFT_HOME=${FROZEN_STACK}/ms-swift
MEGATRON_PATH=${FROZEN_STACK}/Megatron-LM
MCORE_BRIDGE_PATH=${FROZEN_STACK}/mcore-bridge/src
ENV_PATH=/path/to/storage/research/miniconda3/envs/swift
AOSS_CONF=/path/to/storage/research/aoss.conf
MODEL_PATH=/path/to/checkpoints/workflow_sft
DATA_DIR=${EXP_DIR}/data/dev_4000_800
TRAIN_JSONL=${DATA_DIR}/train.jsonl
VAL_JSONL=${DATA_DIR}/validation.jsonl
ENTRY=${EXP_DIR}/scripts/critic_weighted_entry.py
CACHE_DIR=${EXP_DIR}/${CACHE_NAME}
OUTPUT_ROOT=/path/to/storage/multimodal_ckpt/research/autotrain/output/critic_singlemodel_r123_16gpu_20260824
OUTPUT_DIR=${OUTPUT_ROOT}/${RUN_NAME}
LOG_DIR=${EXP_DIR}/logs/16gpu/${RUN_NAME}

TP=2
PP=4
CP=1
VPP=2
NPROC_PER_NODE=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
MAX_LENGTH=8192

NODE_RANK=${RANK:-${OMPI_COMM_WORLD_RANK:-0}}
NNODES=${WORLD_SIZE:-$(grep -c slots "${HOSTFILE:-/etc/mpi/hostfile}" 2>/dev/null || echo 1)}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
MASTER_PORT=${MASTER_PORT:-29543}
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
DP=$((TOTAL_GPUS / (TP * PP * CP)))

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/RANK_${NODE_RANK}_${RUN_STAMP}.log
exec > >(tee "${LOG_FILE}") 2>&1

echo "variant=${VARIANT} run_name=${RUN_NAME}"
echo "experiment=${EXP_DIR}"
echo "dependency_lock=ms-swift:b8241e85781b8f8c8f3f4b1019082657aef14603 megatron:31e3277683dbf06cff4732cbc0033573437d1097 mcore-bridge:c527134caa372e85e8dbf75127d4df02b088bd44"
echo "model=${MODEL_PATH}"
echo "loss_mode=${LOSS_MODE} lr=${LR} min_lr=${MIN_LR}"
echo "train=${TRAIN_JSONL} validation=${VAL_JSONL}"
echo "cache=${CACHE_DIR} output=${OUTPUT_DIR}"
echo "framework_env rank=${NODE_RANK} world_size=${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "topology=TP${TP}_PP${PP}_CP${CP}_DP${DP}_VPP${VPP} nodes=${NNODES} total_gpus=${TOTAL_GPUS}"
echo "packing=${PACKING} padding_free=${PADDING_FREE} use_val=${USE_VAL} dataset_shuffle=false repeat_time=1.0 overlap_p2p_comm=${OVERLAP_P2P_COMM} variable_seq_lengths=${VARIABLE_SEQ_LENGTHS}"

if [ "${NNODES}" -ne 2 ] || [ "${TOTAL_GPUS}" -ne 16 ] || [ "${DP}" -ne 2 ]; then
  echo "expected two 8-GPU nodes with DP2; got nodes=${NNODES} total_gpus=${TOTAL_GPUS} DP=${DP}" >&2
  exit 2
fi
for required in \
  "${SWIFT_HOME}" "${ENV_PATH}" "${MODEL_PATH}" "${TRAIN_JSONL}" \
  "${ENTRY}" "${CACHE_DIR}/train/train" "${MEGATRON_PATH}/megatron" \
  "${MCORE_BRIDGE_PATH}/mcore_bridge"; do
  if [ ! -e "${required}" ]; then
    echo "required path missing: ${required}" >&2
    exit 2
  fi
done
if [ "${USE_VAL}" = true ]; then
  for required in "${VAL_JSONL}" "${CACHE_DIR}/validation/train"; do
    if [ ! -e "${required}" ]; then
      echo "required validation path missing: ${required}" >&2
      exit 2
    fi
  done
fi

source "${ENV_PATH%/envs/*}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${SWIFT_HOME}"

export AOSS_CONF
export PYTHONNOUSERSITE=1
export PYTHONPATH=${MCORE_BRIDGE_PATH}:${MEGATRON_PATH}:${SWIFT_HOME}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=${EXP_DIR}/cache/torchinductor_16gpu_rank${NODE_RANK}
export TRITON_CACHE_DIR=${EXP_DIR}/cache/triton_16gpu_rank${NODE_RANK}
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=1
export USE_PYTORCH_MEM_PROFILER=0
export IMAGE_MAX_TOKEN_NUM=2048
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export MAX_PIXELS=200704
export SWIFT_USE_MCORE_GDN=1

echo "running_frozen_import_smoke node_rank=${NODE_RANK}"
"${ENV_PATH}/bin/python" -u "${EXP_DIR}/scripts/import_smoke_frozen.py"
echo "frozen_import_smoke_complete node_rank=${NODE_RANK}"

COMMON_ARGS=(
  --model "${MODEL_PATH}"
  --save_safetensors true
  --load_from_cache_file true
  --loss_scale "${LOSS_MODE}"
  --enable_thinking false
  --add_non_thinking_prefix false
  --response_prefix ""
  --cached_dataset "${CACHE_DIR}/train/train"
  --pipeline_model_parallel_size "${PP}"
  --tensor_model_parallel_size "${TP}"
  --context_parallel_size "${CP}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --packing "${PACKING}"
  --padding_free "${PADDING_FREE}"
  --dataset_shuffle false
  --val_dataset_shuffle false
  --seed 20260824
  --data_seed 20260824
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --virtual_pipeline_model_parallel_size "${VPP}"
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
  --overlap_p2p_comm "${OVERLAP_P2P_COMM}"
  --variable_seq_lengths "${VARIABLE_SEQ_LENGTHS}"
  --report_to tensorboard
  --profile false
  --use_pytorch_profiler false
)

if [ "${USE_VAL}" = true ]; then
  COMMON_ARGS+=(--cached_val_dataset "${CACHE_DIR}/validation/train")
fi

if [ -n "${TRAIN_ITERS}" ]; then
  COMMON_ARGS+=(--train_iters "${TRAIN_ITERS}")
else
  COMMON_ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

if [ "${LOSS_MODE}" = default ]; then
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    swift/cli/_megatron/sft.py "${COMMON_ARGS[@]}"
else
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${ENTRY}" train "${COMMON_ARGS[@]}"
fi

echo "training_complete variant=${VARIANT} output=${OUTPUT_DIR}"
