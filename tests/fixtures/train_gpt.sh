#!/usr/bin/env bash

torchrun --nproc_per_node 8 pretrain_gpt.py \
  --world-size 64 \
  --num-layers 24 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --seq-length 2048 \
  --vocab-size 50257 \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 1 \
  --micro-batch-size 2 \
  --global-batch-size 256 \
  --bf16 \
  --sequence-parallel \
  --swiglu
