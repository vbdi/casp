export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.."

python quantize_llama/hfize_llava.py \
    --quantized_path qunat_path \
    --hf_output_path out_path \
    --llava_model llava_model \
    --compressed_model