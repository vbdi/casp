export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.."
python quantize_llama/hfize_llama.py \
    --quantized_path quant_path  \
    --hf_output_path hf_out_path \
    --no_use_cuda_graph \
    --no_use_flash_attn \
    --compressed_model