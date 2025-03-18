export CUDA_VISIBLE_DEVICES=4
cd "$(dirname "$0")/.."

python llava.py  $MODEL_PATH \
    --wbits 2 \
    --tbits 2 \
    --true-sequential \
    --act-order  \
    --groupsize 128 \
    --seqlen 1401 \
    --save_fake model \
    --svd

    
