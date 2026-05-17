#!/bin/bash

export CPU_AFFINITY_CONF=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

IPs=('29.210.144.80' '29.210.144.81' '29.210.144.83' '29.210.144.109')
NPUS_PER_NODE=16
MASTER_ADDR=${IPs[0]}
MASTER_PORT=62301
NNODES=${#IPs[@]}
NODE_RANK=""
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="/apdcephfs_nj11/share_304376610/lsy/ckpts"
DATA_PATH="../data/bookcorpus_ds32_mg/bookcorpus_text_document"
TOKENIZER_PATH="/home/nvme0/DeepSeek-V3.2/"
CKPT_LOAD_DIR="../ckpts/ckpt_64die_TP2PP4EP16CP1_Layer16"

TP=2
PP=4
EP=16
CP=1
NUM_LAYERS=16
SEQ_LEN=4096
MBS=1
GBS=64

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --transformer-impl transformer_engine \
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --experimental-attention-variant dsa
"

MOE_ARGS="
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq -1 \
    --moe-shared-expert-intermediate-size 2048 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4
"

ROPE_ARGS="
    --rope-type yarn \
    --rope-scaling-factor 40 \
    --mscale 1.0 \
    --mscale-all-dim 1.0
"

MEM_ARGS="
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --swap-optimizer
"

GPT_ARGS="
    --kv-channels 64 \
    --no-rope-fusion \
    --use-flash-attn \
    --use-distributed-optimizer \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --swiglu \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10368 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $MEM_ARGS \
    --distributed-backend nccl \
    --exit-interval 50000 \
    --load $CKPT_LOAD_DIR \
    --ckpt-format torch \
    --save $CKPT_SAVE_DIR
