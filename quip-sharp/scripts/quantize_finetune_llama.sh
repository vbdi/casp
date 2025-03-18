export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.."
find $CONDA_PREFIX -name nvcc

CKPT=models
HF=hf
LOG=logs
HESS=hessians

python -m quantize_llama.quantize_finetune_llama \
    --save_path $CKPT/llama-2-7B-qk0.25E8P12RVQ3B-1to7and31-E8P12RVQ3B-8to30EP12 \
    --codebook E8P12  \
    --codebook_qk E8P12RVQ3B \
    --codebook_ov E8P12 \
    --codebook_mlp E8P12 \
    --resid_scale_override_qk 0.9 \
    --resid_scale_override_ov 0.9 \
    --resid_scale_override_mlp 0.9 \
    --skip_layers 0 1 2 3 31  \
    --base_model base_model \
    --hessian_path $HESS/llama-2-7b-0.25qk  \
    --devset_size 384 \
    --ft_valid_size 128 \
    --ft_epochs 0 \
    --compressed_model 
   
