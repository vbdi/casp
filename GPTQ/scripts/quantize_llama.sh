export CUDA_VISIBLE_DEVICES=6
cd "$(dirname "$0")/.."
python llama.py  model  \
    --wbits 2 \
    --true-sequential \
    --act-order  \
    --groupsize 128 \
    --seqlen 4096 \
    --save_fake model 
    
