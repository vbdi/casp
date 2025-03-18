cd "$(dirname "$0")/.."
ratio=0.25 python convert_to_hf_llava.py \
    $BASE_MODEL  \
    $QUANTIZED  \
    $QUANTIZED_HF  \
    --save_safetensors \
    --svd