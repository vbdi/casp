cd "$(dirname "$0")/.."

export NUMEXPR_MAX_THREADS=128
python main_llava.py $MODEL_PATH pajama  \
 --nsamples=128 \
 --tbits=2 \
 --val_size=32 \
 --num_codebooks=2 \
 --nbits_per_codebook=8 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --resume \
 --offload_activations  \
 --save models/llava-1.5-7b-hf-AQLM0.25qk-2bit-128samples \
 --svd 

