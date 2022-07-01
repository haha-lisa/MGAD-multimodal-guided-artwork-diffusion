# MGAD-multimodal-guided-artwork-diffusion

## Official Pytorch implementation of "Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion" (ACM Multimedia 2022 Accepted)

![MAIN3_e2-min](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion/blob/main/teaser.jpg)


## Environment
* Pytorch 1.9.0, Python 3.9
* NVIDIA A40
* 512 defaults: 43 GB

```
conda create -n mgad python=3.9
conda activate mgad
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install dependencies
```
git clone https://github.com/openai/CLIP
git clone https://github.com/openai/guided-diffusion
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips
```

## Download the diffusion model
```
curl -OL --http1.1 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
```
## Download the model checkpoint
A 481M parameter unconditional model trained by Katherine Crowson(https://twitter.com/RiversHaveWings)
on a 33 million image original resolution subset of Yahoo Flickr Creative Commons 100 Million.
```
curl -OL --http1.1 'https://the-eye.eu/public/AI/models/v-diffusion/yfcc_1.pth'
```

## Run
```
python mgad.py -p "A stunning natural landscape painting is created by an artist Paul Cezanne in post-impressionism style." --image_prompts "./image_prompts/1.jpg" -t 2000 -ds 2000 -tvs 300 -o "./results/PC-landscape/PC-landscape"
```

## Cite
```
@inproceedings{Huang:2022:MGAD,
author = {Nisha Huang, Fan Tang, Weiming Dong, Changsheng Xu},
title = {Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion},
year = {2022},
booktitle = {The 30th ACM International Conference on Multimedia (ACM MM'22)},
}
```
