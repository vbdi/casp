<div align ="center">

<h2>CASP: Compression of Large Multimodal Models Based on Attention Sparsity</h2>

[Mohsen Gholami](https://scholar.google.ca/citations?hl=en&user=6zlnAJgAAAAJ&view_op=list_works&sortby=pubdate), [Mohammad Akabri](https://scholar.google.ca/citations?user=-88eCXIAAAAJ&hl=en), [Kevin Cannons](https://scholar.google.com/citations?user=2JtzQe4AAAAJ&hl=en), [Yong Zhang](https://scholar.google.ca/citations?user=K2zamrwAAAAJ&hl=en),

Huawei Technologies Canada

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/pdf/2503.05936)

</div>




The code will be released soon...
<p float="left">
  <img src="assets/teaser.png" alt="Image 1" width="38%" />
  <img src="assets/figure_proof.png" alt="Image 2" width="58%" />
</p>

##  Highlights

* CASP proposes a 2-bit compression method for VLMs that is compatible with any quantization technique and enhances state-of-the-art 2-bit quantization methods (AQLM and QuIP#) by an average of 21% on image- and video-language benchmarks


##  Scripts
### CASP<sub>QuIP\#</sub> :

Follow the below steps to prepare CASP<sub>QuIP\#</sub>

1. To prepare llava with low-rank compressed W<sub>q</sub> and W<sub>k</sub> :

    ```bash 
    SVD/scripts/svd_llava.sh
    ```

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


## 📚 Citation
If you find CASP useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.
```bibtex
@misc{gholami2025caspcompressionlargemultimodal,
      title={CASP: Compression of Large Multimodal Models Based on Attention Sparsity}, 
      author={Mohsen Gholami and Mohammad Akbari and Kevin Cannons and Yong Zhang},
      year={2025},
      eprint={2503.05936},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.05936}, 
}
```
