export CUDA_VISIBLE_DEVICES=7
cd "$(dirname "$0")/.."
python convert_to_hf.py \
    llama-2-7b-hf \
    CASP_AQLM  \
    CASP_AQLM_hf  \
    --save_safetensors