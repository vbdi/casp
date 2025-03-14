# CASP: Compression of Large Multimodal Models Based on Attention Sparsity
The code will be released soon...
<p float="left">
  <img src="assets/teaser.png" alt="Image 1" width="38%" />
  <img src="assets/figure_proof.png" alt="Image 2" width="58%" />
</p>

### CASP<sub>QuIP\#</sub> :

Follow the below steps to prepare CASP<sub>QuIP\#</sub>

1. To prepare llava with low-rank compressed W<sub>q</sub> and W<sub>k</sub> :

    ` bash SVD/scripts/svd_llava.sh`

2. To prepare hessians for QuIP\#:

    ` bash quip-sharp/scripts/hfize_llava.sh `

3. Quantization:

    ` bash quip-sharp/scripts/quantize_finetune_llava.sh `

### CASP<sub>AQLM</sub> :

Follow the below steps to prepare CASP<sub>AQLM</sub>

1. To prepare llava with low-rank compressed W<sub>q</sub> and W<sub>k</sub> :

    ` bash SVD/scripts/svd_llava.sh`
    
2. Quantization:

    ` bash AQLM/scripts/quantize_llava.sh `


### CASP<sub>GPTQ</sub> :

Follow the below steps to prepare CASP<sub>GPTQ</sub>

1. To prepare llava with low-rank compressed W<sub>q</sub> and W<sub>k</sub> :

    ` bash SVD/scripts/svd_llava.sh`

2. Quantization:

    ` bash GPTQ/scripts/quantize_llava.sh`