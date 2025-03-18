export CUDA_VISIBLE_DEVICES=5
cd "$(dirname "$0")/.."
find $CONDA_PREFIX -name nvcc

CKPT=models
HF=hf
LOG=logs
HESS=hessians

python -m quantize_llama.quantize_finetune_llava \
    --save_path $CKPT \
    --codebook E8P12 \
    --codebook_qk E8P12 \
    --codebook_ov E8P12 \
    --codebook_mlp E8P12 \
    --resid_scale_override_qk 0.9 \
    --resid_scale_override_ov 0.9 \
    --resid_scale_override_mlp 0.9 \
    --skip_layers 0 1 2 3 4 31  \
    --base_model   \
    --hessian_path $HESS/LLaVA_NeXT_Video_7B_hf_whitening_only_0.5qk_proj \
    --devset_size 384 \
    --ft_valid_size 128 \
    --compressed_model \
    --ft_epochs 0 
    
    #>> $LOG/llama_7b_whitening_only_0.2qk0.7ov_qk_proj_ov_proj_2bit 2>&1