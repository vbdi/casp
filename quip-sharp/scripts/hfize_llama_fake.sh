export CUDA_VISIBLE_DEVICES=0

python quantize_llama/hfize_llama_fake.py \
    --quantized_path quantize_path \
    --hf_output_path output \
    --model_base base_model \
    --no_use_cuda_graph \
    --no_use_flash_attn \
    --compressed_model