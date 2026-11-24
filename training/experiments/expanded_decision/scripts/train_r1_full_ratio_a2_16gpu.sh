#!/bin/bash
set -euo pipefail

VARIANT=${1:?usage: train_r1_full_ratio_16gpu.sh VARIANT}
case "${VARIANT}" in
  a2_balanced_contains_r1) RUN_NAME=r1_full_a2_balanced_contains_r1_13026_defaultloss_lr1e6 ;;
  a_balanced) RUN_NAME=r1_full_a_balanced_13026_defaultloss_lr1e6 ;;
  b_natural_ratio_controlled) RUN_NAME=r1_full_b_ratio215_13026_defaultloss_lr1e6 ;;
  c_all_unique) RUN_NAME=r1_full_c_all20501_defaultloss_lr1e6 ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

EXP_DIR=/path/to/experiments/expanded_decision
SOURCE_EXP=/path/to/experiments/decision_sft
FROZEN_STACK=${SOURCE_EXP}/deps/frozen_b8241e8_mg31e327_mb_c527134
SWIFT_HOME=${FROZEN_STACK}/ms-swift
MEGATRON_PATH=${FROZEN_STACK}/Megatron-LM
MCORE_BRIDGE_PATH=${FROZEN_STACK}/mcore-bridge/src
ENV_PATH=/path/to/storage/research/miniconda3/envs/swift
AOSS_CONF=/path/to/storage/research/aoss.conf
MODEL_PATH=/path/to/checkpoints/workflow_sft
DATA_DIR=${EXP_DIR}/data/${VARIANT}
CACHE_DIR=${EXP_DIR}/cache/${VARIANT}
OUTPUT_ROOT=/path/to/storage/multimodal_ckpt/research/autotrain/output/critic_r1_full_ratio_abc_16gpu_20260826
OUTPUT_DIR=${OUTPUT_ROOT}/${RUN_NAME}
LOG_DIR=${EXP_DIR}/logs/${RUN_NAME}
IMPORT_SMOKE=${SOURCE_EXP}/scripts/import_smoke_frozen.py

TP=2
PP=4
CP=1
VPP=2
NPROC_PER_NODE=${GPUS_PER_NODE:-8}
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-${RANK:-${OMPI_COMM_WORLD_RANK:-0}}}}
NNODES=${SENSECORE_PYTORCH_NNODES:-${NNODES:-2}}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
MASTER_PORT=${MASTER_PORT:-29543}
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
DP=$((TOTAL_GPUS / (TP * PP * CP)))

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
exec > >(tee "${LOG_DIR}/bootstrap_rank_${NODE_RANK}_${RUN_STAMP}.log") 2>&1

echo "variant=${VARIANT} run_name=${RUN_NAME}"
echo "model=${MODEL_PATH}"
echo "data=${DATA_DIR} cache=${CACHE_DIR} output=${OUTPUT_DIR}"
echo "loss=default lr=1e-6 repeat_time=1.0 epoch=1"
echo "topology=TP${TP}_PP${PP}_CP${CP}_DP${DP}_VPP${VPP} nodes=${NNODES} total_gpus=${TOTAL_GPUS}"

if [ "${NNODES}" -ne 2 ] || [ "${TOTAL_GPUS}" -ne 16 ] || [ "${DP}" -ne 2 ]; then
  echo "TOPOLOGY_REJECT expected two 8-GPU nodes with DP2; nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} dp=${DP}" >&2
  exit 2
fi
for required in \
  "${SWIFT_HOME}" "${ENV_PATH}" "${MODEL_PATH}" "${DATA_DIR}/train.jsonl" \
  "${DATA_DIR}/validation.jsonl" "${CACHE_DIR}/train/train" \
  "${CACHE_DIR}/validation/train" "${MEGATRON_PATH}/megatron" \
  "${MCORE_BRIDGE_PATH}/mcore_bridge" "${IMPORT_SMOKE}"; do
  if [ ! -e "${required}" ]; then
    echo "required path missing: ${required}" >&2
    exit 2
  fi
done

source "${ENV_PATH%/envs/*}/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
cd "${SWIFT_HOME}"
export AOSS_CONF PYTHONNOUSERSITE=1
export PYTHONPATH=${MCORE_BRIDGE_PATH}:${MEGATRON_PATH}:${SWIFT_HOME}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=${EXP_DIR}/runtime_cache/torchinductor_${VARIANT}_rank${NODE_RANK}
export TRITON_CACHE_DIR=${EXP_DIR}/runtime_cache/triton_${VARIANT}_rank${NODE_RANK}
export TORCH_COMPILE_DISABLE=0 TORCHDYNAMO_DISABLE=1 USE_PYTORCH_MEM_PROFILER=0
export IMAGE_MAX_TOKEN_NUM=2048 VIDEO_MAX_TOKEN_NUM=128 FPS_MAX_FRAMES=16 MAX_PIXELS=200704
export SWIFT_USE_MCORE_GDN=1

"${ENV_PATH}/bin/python" -u "${IMPORT_SMOKE}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  swift/cli/_megatron/sft.py \
  --model "${MODEL_PATH}" \
  --save_safetensors true \
  --load_from_cache_file true \
  --loss_scale default \
  --enable_thinking false \
  --add_non_thinking_prefix false \
  --response_prefix "" \
  --cached_dataset "${CACHE_DIR}/train/train" \
  --cached_val_dataset "${CACHE_DIR}/validation/train" \
  --pipeline_model_parallel_size 4 \
  --tensor_model_parallel_size 2 \
  --context_parallel_size 1 \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --packing true \
  --padding_free true \
  --dataset_shuffle false \
  --val_dataset_shuffle false \
  --seed 20260824 \
  --data_seed 20260824 \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --virtual_pipeline_model_parallel_size "${VPP}" \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --cross_entropy_fusion_impl te \
  --packing_cp_align false \
  --lr 1e-6 \
  --lr_warmup_fraction 0.2 \
  --min_lr 1e-7 \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps 200 \
  --logging_steps 1 \
  --save_steps 200 \
  --max_length 8192 \
  --dataloader_num_workers 1 \
  --no_save_optim true \
  --no_save_rng true \
  --dataset_num_proc 1 \
  --sequence_parallel true \
  --attention_backend flash \
  --strict false \
  --truncation_strategy delete \
  --vit_gradient_checkpointing false \
  --freeze_llm false \
  --freeze_vit true \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --overlap_p2p_comm true \
  --variable_seq_lengths true \
  --report_to tensorboard \
  --profile false \
  --use_pytorch_profiler false \
  --num_train_epochs 1

echo "training_complete variant=${VARIANT} output=${OUTPUT_DIR}"
