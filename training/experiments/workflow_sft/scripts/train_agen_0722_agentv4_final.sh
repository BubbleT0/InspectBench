#!/bin/bash
# Qwen3.6-27B 8N training — sft_ags_only_v1.0, 3 epochs, max_length=65536.
# Adapted from qwen36_27b_train_multi_node_v2.sh.

# =====================================================================
# >>>>>>>>>>>>>> 初始使用修改区 <<<<<<<<<<<<
# =====================================================================

ENV_PATH=/path/to/storage/research/miniconda3/envs/swift
AOSS_CONF=/path/to/storage/research/aoss.conf

CACHE_BASE_DIR=/path/to/storage/multimodal_ckpt/research/autotrain/cache/data
USER_CACHE_DIR=/path/to/storage/multimodal_ckpt/research/autotrain/cache/node_own
SWIFT_HOME=/path/to/storage/multimodal_ckpt/research/autotrain/ms-swift

model_path=/path/to/storage/multimodal_ckpt/model/Qwen3.6-27B
model_name=$(basename "$model_path")
echo "model: $model_name"

micro_batch_size=1
num_train_epochs=3
lr=2e-5
lr_warmup_fraction=0.10
min_lr=1e-6

TP=2
PP=4
EP=1
CP=4
VPP=2
ETP=1

save_steps=15
global_batch_size=64
max_length=131072

# =====================================================================
# >>>>>>>>>>>>>> 实验/数据 <<<<<<<<<<<<<<<<<<<<<<<<
# =====================================================================

exp_name="${model_name}_TP${TP}_CP${CP}_PP${PP}_VPP${VPP}_EP${EP}_ETP${ETP}_mbs${micro_batch_size}_${max_length}${EXP_SUFFIX:+_${EXP_SUFFIX}}"

resume_flag=false
resume_dir=""

CACHE_DIR="${CACHE_DIR_OVERRIDE:-${CACHE_BASE_DIR}/sft_ags_v1_0608}"
data_dir="${DATA_DIR_OVERRIDE:-/path/to/storage/multimodal_data/research/data/0608/sft_ags_only_v1.0/meta_msgs/meta}"
use_data_json=()
for data_path in "${data_dir}"/*.jsonl; do
    [ -f "${data_path}" ] || continue
    use_data_json+=("${data_path}")
done

# =====================================================================
# >>>>>>>>>>>>>> repeat_map.tsv 载入 (注入 #N 到 packing) <<<<<<<<<<<<<<
# =====================================================================
REPEAT_MAP_TSV="${REPEAT_MAP_TSV:-${SWIFT_HOME}/shell/scripts/repeat_map_ags_v1.tsv}"
declare -A SHARD_N
if [ -f "${REPEAT_MAP_TSV}" ]; then
    while IFS=$'\t' read -r shard_name jsonl_path N orig_length repeat_time; do
        [ "${shard_name}" = "shard_name" ] && continue
        SHARD_N["${shard_name}"]="${N}"
    done < "${REPEAT_MAP_TSV}"
    echo "[repeat_map] loaded ${#SHARD_N[@]} entries from ${REPEAT_MAP_TSV}"
else
    echo "[repeat_map] NO TSV at ${REPEAT_MAP_TSV} — falling back to full-length sampling"
fi

# =====================================================================
# >>>>>>>>>>>>>> Resume 模式 <<<<<<<<<<<<<<<<<<<<<<
# =====================================================================

RESUME_DIR=""
if [ "$resume_flag" = "resume_enabled" ]; then
    SAVE_SAFETENSORS=false; NO_SAVE_OPTIM=false; NO_SAVE_RNG=false; FINETUNE_FLAG=true
elif [ "$resume_flag" = "resume" ] && [ -n "$resume_dir" ]; then
    SAVE_SAFETENSORS=false; NO_SAVE_OPTIM=false; NO_SAVE_RNG=false; FINETUNE_FLAG=false
    RESUME_DIR="--load $resume_dir"
else
    SAVE_SAFETENSORS=true; NO_SAVE_OPTIM=true; NO_SAVE_RNG=true; FINETUNE_FLAG=true
fi

# =====================================================================
# >>>>>>>>>>>>>> 一般不动区域 <<<<<<<<<<<<<
# =====================================================================

NPROC_PER_NODE=8
NODE_RANK="${RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
NNODES="${WORLD_SIZE:-$(grep -c 'slots' "${HOSTFILE:-/etc/mpi/hostfile}" 2>/dev/null || echo 1)}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "FRAMEWORK_ENV_DEBUG RANK=${RANK:-unset} WORLD_SIZE=${WORLD_SIZE:-unset} OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-unset} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} HOSTFILE=${HOSTFILE:-unset}"
echo "RESOLVED NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export OMP_NUM_THREADS=1
export AOSS_CONF="${AOSS_CONF}"
export PYTHONPATH=${SWIFT_HOME}/dependencies/Megatron-LM:$PYTHONPATH
export PYTHONPATH=${SWIFT_HOME}:$PYTHONPATH
export USE_PYTORCH_MEM_PROFILER=0

source "${ENV_PATH%/envs/*}/bin/activate"
conda activate "${ENV_PATH}"
which python

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=${USER_CACHE_DIR}/torchinductor_cache/node_rank_${NODE_RANK}
export TRITON_CACHE_DIR=${USER_CACHE_DIR}/triton_cache/node_rank_${NODE_RANK}
export TORCH_COMPILE_DISABLE=0
export TORCHDYNAMO_DISABLE=1

export IMAGE_MAX_TOKEN_NUM=8192
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export MAX_PIXELS=200704

export SWIFT_USE_MCORE_GDN=1

cd "${SWIFT_HOME}"
current_dir=$(pwd)
output=/path/to/storage/multimodal_ckpt/research/autotrain/output

LOG_SUB_FOLDER="${LOG_SUB_FOLDER:-/path/to/storage/multimodal_ckpt/research/autotrain/logs/${exp_name}_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${LOG_SUB_FOLDER}/RANK_${NODE_RANK}.log"
mkdir -p "${LOG_SUB_FOLDER}"
echo "日志将输出到: ${LOG_FILE}"
if command -v tee >/dev/null 2>&1; then
    exec > >(tee "${LOG_FILE}") 2>&1
else
    exec > "${LOG_FILE}" 2>&1
fi

# =====================================================================
# >>>>>>>>>>>>>> Step 1: 预 Packing <<<<<<<<<<<<
# =====================================================================
PACK_NUM_PROC_PER_SHARD="${PACK_NUM_PROC_PER_SHARD:-8}"
PACK_PARALLEL_SHARDS="${PACK_PARALLEL_SHARDS:-8}"
SHARDS_DIR="${CACHE_DIR}/train"
mkdir -p "${SHARDS_DIR}"

total_shards=$(ls "${data_dir}"/*.jsonl 2>/dev/null | wc -l)
done_shards=$(ls -d "${SHARDS_DIR}"/*/train 2>/dev/null | wc -l)
echo "total_shards=${total_shards} done_shards=${done_shards}"

if [ "${SKIP_PACKING:-0}" = "1" ]; then
    echo "[预Packing] SKIP_PACKING=1，直接进入训练（用现有 ${done_shards} shard cache）"
elif [ "${total_shards}" -gt 0 ] && [ "${done_shards}" -eq "${total_shards}" ]; then
    echo "[预Packing] 缓存完整 (${done_shards}/${total_shards})，跳过"
else
    rm -f "${SHARDS_DIR}/.node_${NODE_RANK}_failed"

    my_shards=()
    shard_idx=0
    for data_path in "${data_dir}"/*.jsonl; do
        [ -f "${data_path}" ] || continue
        if [ $((shard_idx % NNODES)) -eq "${NODE_RANK}" ]; then
            my_shards+=("${data_path}")
        fi
        shard_idx=$((shard_idx + 1))
    done
    my_total=${#my_shards[@]}
    echo "[预Packing] NODE ${NODE_RANK}/${NNODES}: 负责 ${my_total} shard / 总 ${total_shards}, parallel=${PACK_PARALLEL_SHARDS} procs/shard=${PACK_NUM_PROC_PER_SHARD}"

    pids=()
    launched=0
    for data_path in "${my_shards[@]}"; do
        shard_name=$(basename "${data_path}" .jsonl)
        shard_out="${SHARDS_DIR}/${shard_name}"
        [ -d "${shard_out}/train" ] && continue
        while [ "$(jobs -rp | wc -l)" -ge "${PACK_PARALLEL_SHARDS}" ]; do sleep 5; done
        launched=$((launched + 1))
        sleep $((1 + RANDOM % 5))
        N="${SHARD_N[${shard_name}]:-}"
        if [ -n "${N}" ]; then
            ds_arg="${data_path}#${N}"
        else
            ds_arg="${data_path}"
        fi
        python3 -u swift/cli/export.py \
            --model "${model_path}" \
            --dataset "${ds_arg}" \
            --split_dataset_ratio 0 \
            --dataset_num_proc "${PACK_NUM_PROC_PER_SHARD}" \
            --to_cached_dataset true \
            --truncation_strategy delete \
            --output_dir "${shard_out}" \
            --max_length "${max_length}" \
            > "${LOG_SUB_FOLDER}/pack_${shard_name}.log" 2>&1 &
        pids+=($!)
    done
    echo "[预Packing] NODE ${NODE_RANK} launched ${launched} shard, waiting..."

    failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$((failed + 1))
    done
    if [ "${failed}" -ne 0 ]; then
        touch "${SHARDS_DIR}/.node_${NODE_RANK}_failed"
        echo "[预Packing] NODE ${NODE_RANK} ${failed} shard 失败，sentinel 已写"
    else
        echo "[预Packing] NODE ${NODE_RANK} 本节点 packing 完成"
    fi
    sync 2>/dev/null || true

    echo "[预Packing] NODE ${NODE_RANK} 进入全局 barrier..."
    while true; do
        cached=$(ls -d "${SHARDS_DIR}"/*/train 2>/dev/null | wc -l)
        any_failed=$(ls "${SHARDS_DIR}"/.node_*_failed 2>/dev/null | wc -l)
        if [ "${any_failed}" -gt 0 ]; then
            echo "[预Packing] 检测到 ${any_failed} 个 fail sentinel，全节点退出"
            exit 1
        fi
        if [ "${cached}" -ge "${total_shards}" ]; then
            break
        fi
        echo "[预Packing] NODE ${NODE_RANK} barrier: cached=${cached}/${total_shards}"
        sleep 30
    done
    echo "[预Packing] NODE ${NODE_RANK} 所有节点 packing 完成，进入训练"
fi

if [ "${PACKING_ONLY:-0}" = "1" ]; then
    echo "[PACKING_ONLY=1] packing 完成, 退出, 不进入训练"
    exit 0
fi

cached_dataset_args=()
skip_tools=0
keep_count=0
for shard_path in "${SHARDS_DIR}"/*/; do
    [ -d "${shard_path}train" ] || continue
    if [ "${SCHEMA_FILTER:-no_tools}" = "no_tools" ]; then
        info_file="${shard_path}train/dataset_info.json"
        if [ -f "${info_file}" ] && python3 -c "import json,sys;d=json.load(open('${info_file}'));sys.exit(0 if 'tools' in d.get('features',{}) else 1)" 2>/dev/null; then
            skip_tools=$((skip_tools + 1))
            continue
        fi
    fi
    cached_dataset_args+=("${shard_path}train")
    keep_count=$((keep_count + 1))
done
echo "cached shards count: ${keep_count} (skipped ${skip_tools} with tools field, filter=${SCHEMA_FILTER:-no_tools})"

echo "start torchrun with node rank ${NODE_RANK}"

# =====================================================================
# >>>>>>>>>>>>>> Step 2: Megatron 训练 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =====================================================================

torchrun \
    --nproc_per_node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    swift/cli/_megatron/sft.py \
    --model ${model_path} \
    $RESUME_DIR \
    --save_safetensors ${SAVE_SAFETENSORS} \
    --load_from_cache_file true \
    --loss_scale ignore_empty_think \
    --cached_dataset "${cached_dataset_args[@]}" \
    --pipeline_model_parallel_size ${PP} \
    --tensor_model_parallel_size ${TP} \
    --context_parallel_size ${CP} \
    --micro_batch_size ${micro_batch_size} \
    --global_batch_size ${global_batch_size} \
    --packing true \
    --padding_free true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --virtual_pipeline_model_parallel_size $VPP \
    --num_train_epochs ${num_train_epochs} \
    --finetune ${FINETUNE_FLAG} \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --packing_cp_align true \
    --lr ${lr} \
    --lr_warmup_fraction ${lr_warmup_fraction} \
    --min_lr ${min_lr} \
    --output_dir ${output}/${exp_name} \
    --eval_steps 2000000000 \
    --logging_steps 1 \
    --save_steps ${save_steps} \
    --max_length ${max_length} \
    --dataloader_num_workers 1 \
    --no_save_optim ${NO_SAVE_OPTIM} \
    --no_save_rng ${NO_SAVE_RNG} \
    --dataset_num_proc 1 \
    --sequence_parallel true \
    --attention_backend flash \
    --strict false \
    --truncation_strategy delete \
    --vit_gradient_checkpointing false \
    --freeze_llm false \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --profile false \
    --use_pytorch_profiler false \
    --report_to tensorboard
