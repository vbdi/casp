#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.."

python svd_llava.py \
    --model  llava-hf/llava-1.5-7b-hf \
    --ratios 0.75  \
    --dataset rp \
    --whitening_nsamples 256 \
    --seed 3 \
    --save_path models 
