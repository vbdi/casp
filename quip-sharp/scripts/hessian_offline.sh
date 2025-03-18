export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")/.."
python quantize_llama/hessian_offline_llama.py \
    --base_model base_model \
    --save_path save_path \
    --compressed_model


    