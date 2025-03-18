export CUDA_VISIBLE_DEVICES=2
cd "$(dirname "$0")/.."
python quantize_llama/hessian_offline_llava.py \
    --base_model basemodel \
    --save_path save_path \
    --compressed_model


    