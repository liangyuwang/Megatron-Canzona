#!/bin/bash
# ============================================================
# Megatron-LM GPT Pretraining Script (NTP)
# ============================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# ============================================================
# Basic configuration
# ============================================================
EXP_NAME=${EXP_NAME:-"GPT-NTP-wikitext103"}
MODEL_SIZE=${MODEL_SIZE:-"1B"}
OUTPUT_CHECKPOINT_PATH="./exp/gpt_${MODEL_SIZE}/${EXP_NAME}/"
OUTPUT_TENSORBOARD_PATH="./exp/gpt_${MODEL_SIZE}/${EXP_NAME}/"

DATA_PATH="./data/wikitext103_text_document"

VOCAB_FILE="./data/vocab/gpt2-vocab.json"
MERGE_FILE="./data/vocab/gpt2-merges.txt"

DATA_CACHE_PATH="./data/cache"
mkdir -p ${DATA_CACHE_PATH}

SEQ_LEN=2048
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
MP_SIZE=${MP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
GPUS_PER_NODE=8

# ============================================================
# Model size configuration
# ============================================================
case "$MODEL_SIZE" in
    1B)
        NUM_LAYERS=28
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=6144
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=8
        KV_CHANNELS=128
        ;;
    3B)
        NUM_LAYERS=32
        HIDDEN_SIZE=2560
        FFN_HIDDEN_SIZE=10752
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=8
        KV_CHANNELS=128
        ;;
    8B)
        NUM_LAYERS=36
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=12288
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=8
        KV_CHANNELS=128
        ;;
    14B)
        NUM_LAYERS=40
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=17408
        NUM_ATTN_HEADS=40
        NUM_QUERY_GROUPS=8
        KV_CHANNELS=128
        ;;
    32B)
        NUM_LAYERS=64
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=25600
        NUM_ATTN_HEADS=64
        NUM_QUERY_GROUPS=8
        KV_CHANNELS=128
        ;;
    *)
        echo "Unsupported model size: ${MODEL_SIZE}"
        exit 1
        ;;
esac

# ============================================================
# Training hyperparameters
# ============================================================
LR=${LR:-4e-4}
MIN_LR=${MIN_LR:-4e-5}
TRAIN_STEPS=${TRAIN_STEPS:-500}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-50}

TRAIN_TOKENS=$(( ${TRAIN_STEPS} * ${SEQ_LEN} * ${GLOBAL_BATCH_SIZE} ))
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS} / ${SEQ_LEN} ))
LR_DECAY_TOKENS=$(( ${TRAIN_STEPS} * ${SEQ_LEN} * ${GLOBAL_BATCH_SIZE} ))
LR_DECAY_SAMPLES=$(( ${LR_DECAY_TOKENS} / ${SEQ_LEN} ))
LR_WARMUP_TOKENS=$(( ${LR_WARMUP_STEPS} * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} ))
LR_WARMUP_SAMPLES=$(( ${LR_WARMUP_TOKENS} / ${SEQ_LEN} ))

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPU_SIZE=$(( $GPUS_PER_NODE * $NNODES ))

LOG_INTERVAL=1
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
ACTIVATION_CHECKPOINT="false"
DEBUG=${DEBUG:-0}

NAME="gpt-bf16-${MODEL_SIZE}-micro${BATCH_SIZE}-gqa${NUM_QUERY_GROUPS}-mp${MP_SIZE}-pp${PP_SIZE}-lr${LR}-bs${GLOBAL_BATCH_SIZE}-gpus${GPU_SIZE}-seq${SEQ_LEN}"
CHECKPOINT_PATH="${OUTPUT_CHECKPOINT_PATH}/${NAME}"
mkdir -p ${CHECKPOINT_PATH}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_TENSORBOARD_PATH}/${NAME}_${current_time}"
RESUME_PATH=${RESUME_PATH:-$CHECKPOINT_PATH}

# ============================================================
# Model arguments
# ============================================================
GPT_ARGS="
        --qk-layernorm \
        --manual-gc \
        --manual-gc-interval 100 \
        --logging-level 20 \
        --use-mcore-models \
        --use-dist-ckpt \
        --auto-detect-ckpt-format \
        --data-cache-path ${DATA_CACHE_PATH} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --group-query-attention \
        --kv-channels ${KV_CHANNELS} \
        --num-query-groups ${NUM_QUERY_GROUPS} \
        --sequence-parallel \
        --disable-bias-linear \
        --use-rotary-position-embeddings \
        --no-position-embedding \
        --hidden-dropout 0 \
        --attention-dropout 0 \
        --use-flash-attn \
        --swiglu \
        --normalization RMSNorm \
        --seed 42 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --norm-epsilon 1e-06 \
        --lr-decay-style cosine \
        --split 949,50,1 \
        --log-interval ${LOG_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 2 \
        --bf16 \
        --load ${RESUME_PATH} \
        --tensorboard-queue-size 500 \
        --tensorboard-log-interval 1 \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --log-throughput \
        --attention-softmax-in-fp32 \
        --no-create-attention-mask-in-dataloader \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --eval-interval 1000 \
        --eval-iters 10 \
"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    GPT_ARGS="${GPT_ARGS} --recompute-granularity selective"
fi

if [ "${DEBUG}" -ne 1 ]; then
    GPT_ARGS="${GPT_ARGS} --save ${CHECKPOINT_PATH}"
fi

# ============================================================
# Data arguments
# ============================================================
DATA_ARGS="
        --tokenizer-type GPT2BPETokenizer \
        --data-path ${DATA_PATH} \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
"

# ============================================================
# Canzona arguments
# ============================================================

# https://arxiv.org/pdf/2602.06079

# ------------------------------------------------------------
# Optimizer selection
# ------------------------------------------------------------
USE_MUON=${USE_MUON:-0}
USE_SOAP=${USE_SOAP:-0}

if [ "${USE_MUON}" -eq 1 ]; then
    unset CUBLAS_WORKSPACE_CONFIG
    GPT_ARGS="${GPT_ARGS} --optimizer muon"
fi
if [ "${USE_SOAP}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} --optimizer soap"
fi

# ------------------------------------------------------------
# Canzona + Distributed optimizer
# ------------------------------------------------------------
USE_DP_ASYNC_OPT=${USE_DP_ASYNC_OPT:-1}
if [ "${USE_DP_ASYNC_OPT}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

USE_DP_BALANCED_OPT=${USE_DP_BALANCED_OPT:-0}
DP_BALANCED_OPT_ALPHA=${DP_BALANCED_OPT_ALPHA:-1.0}
DP_BALANCED_OPT_COST=${DP_BALANCED_OPT_COST:-"numel"}
export DEBUG_DP_BALANCED_OPT=${DEBUG_DP_BALANCED_OPT:-0}

if [ "${USE_DP_BALANCED_OPT}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --use-dp-balanced-opt \
        --dp-balanced-opt-alpha ${DP_BALANCED_OPT_ALPHA} \
        --dp-balanced-opt-cost ${DP_BALANCED_OPT_COST}"
fi

DP_BALANCED_OPT_LOG_VISUALIZATION=${DP_BALANCED_OPT_LOG_VISUALIZATION:-0}
if [ "${DP_BALANCED_OPT_LOG_VISUALIZATION}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --dp-balanced-opt-log-visualization \
        --dp-balanced-opt-log-path ${OUTPUT_CHECKPOINT_PATH}"
fi

# ------------------------------------------------------------
# Canzona + Tensor parallel
# ------------------------------------------------------------
USE_TP_ASYNC_OPT=${USE_TP_ASYNC_OPT:-0}
if [ "${USE_TP_ASYNC_OPT}" -eq 0 ] && ( [ "${USE_MUON}" -eq 1 ] || [ "${USE_SOAP}" -eq 1 ] ); then
    GPT_ARGS="${GPT_ARGS} --use-tp-sync-opt"
fi

TP_BALANCED_OPT_FUSE=${TP_BALANCED_OPT_FUSE:-1}
TP_BALANCED_OPT_FUSE_SPACE=${TP_BALANCED_OPT_FUSE_SPACE:-400}
export DEBUG_TP_FUSE_COMM=${DEBUG_TP_FUSE_COMM:-0}

if [ "${USE_MUON}" -eq 1 ] || [ "${USE_SOAP}" -eq 1 ]; then
    if [ "${TP_BALANCED_OPT_FUSE}" -eq 1 ]; then
        GPT_ARGS="${GPT_ARGS} --tp-balanced-opt-fuse-space ${TP_BALANCED_OPT_FUSE_SPACE}"
    else
        GPT_ARGS="${GPT_ARGS} --no-async-tp-fuse-comm"
    fi
fi

USE_TP_BALANCED_OPT=${USE_TP_BALANCED_OPT:-0}
TP_BALANCED_OPT_COST=${TP_BALANCED_OPT_COST:-"numel"}
export DEBUG_TP_BALANCED_OPT=${DEBUG_TP_BALANCED_OPT:-0}

if [ "${USE_TP_BALANCED_OPT}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --use-tp-balanced-opt \
        --tp-balanced-opt-cost ${TP_BALANCED_OPT_COST}"
fi

TP_BALANCED_OPT_LOG_VISUALIZATION=${TP_BALANCED_OPT_LOG_VISUALIZATION:-0}
if [ "${TP_BALANCED_OPT_LOG_VISUALIZATION}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --tp-balanced-opt-log-visualization \
        --tp-balanced-opt-log-path ${OUTPUT_CHECKPOINT_PATH}"
fi

# ------------------------------------------------------------
# Matrix-based optimizer splits
# ------------------------------------------------------------
USE_MATRIX_BASED_OPTIM_SPLIT=${USE_MATRIX_BASED_OPTIM_SPLIT:-0}
if [ "${USE_MATRIX_BASED_OPTIM_SPLIT}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --matrix-based-optimizer-split-fc1 \
        --matrix-based-optimizer-split-qkv \
        --matrix-based-optimizer-split-qkv-per-head"
fi

# ============================================================
# Layerwise-Muon arguments
# ============================================================

# https://github.com/NVIDIA/Megatron-LM/pull/2241

LAYERWISE_MUON=${LAYERWISE_MUON:-0}
if [ "${LAYERWISE_MUON}" -eq 1 ]; then
    GPT_ARGS="${GPT_ARGS} \
        --optimizer layerwise-muon \
        "
fi

# ============================================================
# Launch training
# ============================================================
echo "=========================================="
echo "Model size  : ${MODEL_SIZE}"
echo "Num GPUs    : ${GPU_SIZE}"
echo "Batch size  : ${GLOBAL_BATCH_SIZE} (micro: ${BATCH_SIZE})"
echo "Seq length  : ${SEQ_LEN}"
echo "Train steps : ${TRAIN_STEPS}"
echo "LR          : ${LR}"
echo "Data path   : ${DATA_PATH}"
echo "Checkpoint  : ${CHECKPOINT_PATH}"
echo "=========================================="

torchrunx ../../pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    --distributed-backend nccl
