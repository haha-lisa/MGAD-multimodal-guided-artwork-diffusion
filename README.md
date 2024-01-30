# MGAD-multimodal-guided-artwork-diffusion

## Official Pytorch implementation of "Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion" (ACM Multimedia 2022 Accepted) [paper](https://arxiv.org/abs/2209.13360) https://arxiv.org/abs/2209.13360

![MAIN3_e2-min](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion/blob/main/teaser.jpg)

**Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion**<br>
**ACM Multimedia 2022**<br>


## Abstract
> Digital art creation is getting more attention in the multimedia community for providing effective engagement of the public with art. Current digital art generation methods usually use single modality inputs as guidance, limiting the expressiveness of the model and the diversity of generated results. To solve this problem, we propose the multimodal guided artwork diffusion (MGAD) model,  a diffusion-based digital artwork generation method that utilizes multimodal prompts as guidance to control the classifier-free diffusion model. Additionally, the contrastive language-image pretraining (CLIP) model is used to unify text and image modalities. However, the semantic content of multimodal prompts may conflict with each other, which leads to a collapse in generating progress. Extensive experimental results on the quality and quantity of the generated digital art paintings confirm the effectiveness of the combination of the diffusion model and multimodal guidance.

## Environment
* Pytorch 1.9.0, Python 3.9
* NVIDIA A40
* 512 defaults: 38 GB

```
conda create -n mgad python=3.9
conda activate mgad
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install dependencies
```
git clone https://github.com/openai/CLIP
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips
```

## Download the diffusion model
```
curl -OL --http1.1 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
```
## Download the model checkpoint
An unconditional model trained by Katherine Crowson(https://twitter.com/RiversHaveWings)
on a 33 million image original resolution subset of Yahoo Flickr Creative Commons 100 Million.
```
curl -OL --http1.1 'https://the-eye.eu/public/AI/models/v-diffusion/yfcc_1.pth'
```

## Run
```
python mgad.py -p "A stunning natural landscape painting is created by an artist Paul Cezanne in post-impressionism style." --image_prompts "./image_prompts/1.jpg" -t 2000 -ds 2000 -tvs 300 -o "./results/PC-landscape/PC-landscape"
```



## Acknowledgments
*This code borrows heavily from [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch) and [CLIP-Guided-Diffusion](https://github.com/nerdyrodent/CLIP-Guided-Diffusion).
We also thank [CLIP](https://github.com/openai/CLIP) and [guided-diffusion](https://github.com/openai/guided-diffusion).*

## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file.<br>
