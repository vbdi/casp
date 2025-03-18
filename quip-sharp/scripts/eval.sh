export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.."
python  eval/eval_ppl.py \
    --no_use_flash_attn \
    --no_use_cuda_graph \
    --hf_path model