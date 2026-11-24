#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/trajectory_r2
SWIFT_HOME=/path/to/storage/multimodal_ckpt/research/autotrain/ms-swift
ENV_PATH=/path/to/storage/research/miniconda3/envs/swift
AOSS_CONF=/path/to/storage/research/aoss.conf
MODEL_PATH=/path/to/checkpoints/workflow_sft
REPEAT_MAP=${EXP_DIR}/data/stage5_swift/repeat_map.tsv
CACHE_DIR=${EXP_DIR}/cache/critic_train_8192_rp2_neo1_edit
SHARDS_DIR=${CACHE_DIR}/train
OUTPUT_DIR=${EXP_DIR}/output/full_16gpu_tp2_pp4_cp1_dp2_rp2_epoch1_gbs8_neo1_edit
LOG_DIR=${EXP_DIR}/logs/neo1_edit

TP=2
PP=4
CP=1
VPP=2
NPROC_PER_NODE=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
MAX_LENGTH=8192
NUM_TRAIN_EPOCHS=1
LR=1e-6
MIN_LR=1e-7
SAVE_STEPS=500
PACK_PARALLEL_SHARDS=2
PACK_NUM_PROC_PER_SHARD=16

mkdir -p "${SHARDS_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
NODE_RANK="${RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
NNODES="${WORLD_SIZE:-$(grep -c 'slots' "${HOSTFILE:-/etc/mpi/hostfile}" 2>/dev/null || echo 1)}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"
MASTER_PORT="${MASTER_PORT:-29527}"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/RANK_${NODE_RANK}_${RUN_STAMP}.log
exec > >(tee "${LOG_FILE}") 2>&1

echo "experiment=${EXP_DIR}"
echo "model=${MODEL_PATH}"
echo "repeat_map=${REPEAT_MAP}"
echo "cache_dir=${CACHE_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "framework_env RANK=${RANK:-unset} WORLD_SIZE=${WORLD_SIZE:-unset} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "topology=TP${TP}_PP${PP}_CP${CP}_VPP${VPP} node_rank=${NODE_RANK} nnodes=${NNODES} total_gpus=$((NNODES * NPROC_PER_NODE))"
echo "epochs=${NUM_TRAIN_EPOCHS} global_batch_size=${GLOBAL_BATCH_SIZE} micro_batch_size=${MICRO_BATCH_SIZE} max_length=${MAX_LENGTH}"

if [ "${NNODES}" -ne 2 ]; then
    echo "expected exactly 2 nodes for the 16-GPU topology, got ${NNODES}" >&2
    exit 2
fi
for required_path in "${SWIFT_HOME}" "${ENV_PATH}" "${MODEL_PATH}" "${REPEAT_MAP}"; do
    if [ ! -e "${required_path}" ]; then
        echo "required path missing: ${required_path}" >&2
        exit 2
    fi
done

source "${ENV_PATH%/envs/*}/bin/activate"
conda activate "${ENV_PATH}"
cd "${SWIFT_HOME}"

export AOSS_CONF="${AOSS_CONF}"
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

declare -a SHARD_NAMES=()
declare -a SHARD_PATHS=()
declare -a SHARD_COUNTS=()
while IFS=$'\t' read -r shard_name jsonl_path sample_n orig_length repeat_time; do
    [ "${shard_name}" = "shard_name" ] && continue
    SHARD_NAMES+=("${shard_name}")
    SHARD_PATHS+=("${jsonl_path}")
    SHARD_COUNTS+=("${sample_n}")
    echo "repeat_entry shard=${shard_name} original=${orig_length} repeat=${repeat_time} effective=${sample_n}"
done < "${REPEAT_MAP}"

if [ "${#SHARD_NAMES[@]}" -ne 4 ]; then
    echo "expected four training shards in repeat map, got ${#SHARD_NAMES[@]}" >&2
    exit 2
fi

rm -f "${SHARDS_DIR}/.node_${NODE_RANK}_failed"
pids=()
launched=0
for index in "${!SHARD_NAMES[@]}"; do
    if [ $((index % NNODES)) -ne "${NODE_RANK}" ]; then
        continue
    fi
    shard_name=${SHARD_NAMES[$index]}
    jsonl_path=${SHARD_PATHS[$index]}
    sample_n=${SHARD_COUNTS[$index]}
    shard_out=${SHARDS_DIR}/${shard_name}
    if [ -d "${shard_out}/train" ]; then
        echo "packing_skip shard=${shard_name} cache_exists=${shard_out}/train"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "${PACK_PARALLEL_SHARDS}" ]; do sleep 5; done
    echo "packing_start shard=${shard_name} dataset=${jsonl_path} sample_n=${sample_n}"
    python3 -u swift/cli/export.py \
        --model "${MODEL_PATH}" \
        --dataset "${jsonl_path}#${sample_n}" \
        --split_dataset_ratio 0 \
        --dataset_num_proc "${PACK_NUM_PROC_PER_SHARD}" \
        --to_cached_dataset true \
        --truncation_strategy delete \
        --max_length "${MAX_LENGTH}" \
        --output_dir "${shard_out}" \
        >"${LOG_DIR}/pack_${shard_name}.log" 2>&1 &
    pids+=("$!")
    launched=$((launched + 1))
done
echo "packing_launched node=${NODE_RANK} shards=${launched} parallel=${PACK_PARALLEL_SHARDS} procs_per_shard=${PACK_NUM_PROC_PER_SHARD}"

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=$((failed + 1))
done
if [ "${failed}" -ne 0 ]; then
    touch "${SHARDS_DIR}/.node_${NODE_RANK}_failed"
    echo "packing_failed node=${NODE_RANK} failed_shards=${failed}" >&2
fi
sync 2>/dev/null || true

while true; do
    cached=$(find "${SHARDS_DIR}" -mindepth 2 -maxdepth 2 -type d -name train | wc -l)
    fail_sentinels=$(find "${SHARDS_DIR}" -maxdepth 1 -type f -name '.node_*_failed' | wc -l)
    if [ "${fail_sentinels}" -gt 0 ]; then
        echo "packing barrier detected ${fail_sentinels} failure sentinel(s)" >&2
        exit 3
    fi
    if [ "${cached}" -eq "${#SHARD_NAMES[@]}" ]; then
        break
    fi
    echo "packing_barrier cached=${cached}/${#SHARD_NAMES[@]}"
    sleep 20
done

cached_dataset_args=()
for shard_name in "${SHARD_NAMES[@]}"; do
    cache_path=${SHARDS_DIR}/${shard_name}/train
    if [ ! -d "${cache_path}" ]; then
        echo "required cache missing after barrier: ${cache_path}" >&2
        exit 3
    fi
    cached_dataset_args+=("${cache_path}")
done
echo "packing_complete cached_shards=${#cached_dataset_args[@]}"

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
    --loss_scale ignore_empty_think \
    --cached_dataset "${cached_dataset_args[@]}" \
    --pipeline_model_parallel_size "${PP}" \
    --tensor_model_parallel_size "${TP}" \
    --context_parallel_size "${CP}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --packing true \
    --padding_free true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --virtual_pipeline_model_parallel_size "${VPP}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --packing_cp_align false \
    --lr "${LR}" \
    --lr_warmup_fraction 0.2 \
    --min_lr "${MIN_LR}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_steps 2000000000 \
    --logging_steps 1 \
    --save_steps "${SAVE_STEPS}" \
    --max_length "${MAX_LENGTH}" \
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
    --profile false \
    --use_pytorch_profiler false \
    --report_to tensorboard

echo "formal one-epoch training command completed"
