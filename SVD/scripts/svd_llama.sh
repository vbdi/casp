#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")/.."
python svd_llama.py \
    --model  llama-2-7b-hf \
    --ratios 0.75 \
    --seed 3 \
    --model_seq_len 4096 \
    --save_path output \
    --dataset rp \
