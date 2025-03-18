python llava.py  $MODEL_PATH  \
    --wbits 2 \
    --tbits 2 \
    --true-sequential \
    --act-order  \
    --groupsize 128 \
    --seqlen 831 \
    --save_fake model \
    --nsamples 128 \
    --svd

    
