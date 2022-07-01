
from dataclasses import dataclass
from functools import partial
import argparse
import gc
import io
import math
import sys
from pathlib import Path

from IPython import display
import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import os


sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

torch.backends.cudnn.enabled = False



diffc_parser = argparse.ArgumentParser(description='Image generation using Diffusion+CLIP') 

diffc_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default="", dest='prompts')
diffc_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts') 
diffc_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
diffc_parser.add_argument("-st",   "--skip_steps", type=int, help="Skip steps for init image (200-500)", default=0, dest='skip_timesteps') 
diffc_parser.add_argument("-is",   "--init_scale", type=int, help="Initial image scale (e.g. 1000)", default=0, dest='init_scale') 
diffc_parser.add_argument("-bs",   "--batch_size", type=int, help="Batch size", default=1, dest='batch_size')
diffc_parser.add_argument("-nb",   "--num_batches", type=int, help="Number of batches", default=1, dest='n_batches')
diffc_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=16, dest='cutn')
diffc_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=0.5, dest='cut_pow')
diffc_parser.add_argument("-cgs",  "--clip_scale", type=int, help="CLIP guidance scale", default=5000, dest='clip_guidance_scale') 
diffc_parser.add_argument("-tvs",  "--tv_scale", type=float, help="Smoothness scale", default=400, dest='tv_scale') 
diffc_parser.add_argument("-rgs",  "--range_scale", type=int, help="RGB range scale", default=0, dest='range_scale') 
diffc_parser.add_argument("-s",    "--seed", type=int, help="Seed", default=0, dest='seed')
diffc_parser.add_argument("-o",    "--output", type=str, help="Output file", default="output", dest='output')
diffc_parser.add_argument( "-freq","--save_frequency", type=int, default=100, help="Save frequency")
diffc_parser.add_argument("-m",    "--model", type=int, help="model tag", default=1, dest='secondary_model_ver')
diffc_parser.add_argument("-ds",   "--diffusion_steps", type=int, help="Diffusion steps", default=2000, dest='diffusion_steps')
diffc_parser.add_argument("-t",    "--timesteps", type=str, help="Number of timesteps", default='2000', dest='timesteps') 


args = diffc_parser.parse_args()
prompts = [phrase.strip() for phrase in args.prompts.split("|")]
if args.image_prompts:
    args.image_prompts = args.image_prompts.split("|")
    args.image_prompts = [image.strip() for image in args.image_prompts]   

prompts = [phrase.strip() for phrase in args.prompts.split("|")]
image_prompts = args.image_prompts
init_image = args.init_image  
skip_timesteps = args.skip_timesteps 
init_scale = args.init_scale 
batch_size = args.batch_size
n_batches = args.n_batches
cutn = args.cutn
cut_pow = args.cut_pow
clip_guidance_scale = args.clip_guidance_scale 
tv_scale = args.tv_scale
range_scale = args.range_scale 
seed = args.seed
output = args.output
save_frequency = args.save_frequency
secondary_model_ver=args.secondary_model_ver

Path(output).mkdir(parents=True, exist_ok=True)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])




def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)
    
class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.ReLU(inplace=True) if dropout_last else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class yyfc(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 512, 512)
        c = 128
        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(   # 512x512
            ResConvBlock(3 + 16, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 512x512 -> 256x256
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                SkipBlock([
                    nn.AvgPool2d(2),  # 256x256 -> 128x128
                    ResConvBlock(c, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 128x128 -> 64x64
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        SkipBlock([
                            nn.AvgPool2d(2),  # 64x64 -> 32x32
                            ResConvBlock(c * 2, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            SkipBlock([
                                nn.AvgPool2d(2),  # 32x32 -> 16x16
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                SkipBlock([
                                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                                    ResConvBlock(c * 4, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    SkipBlock([
                                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        nn.Upsample(scale_factor=2, mode='bilinear',#4x4->8x8
                                                    align_corners=False),
                                    ]),
                                    ResConvBlock(c * 16, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 4),
                                    SelfAttention2d(c * 4, c * 4 // 64),
                                    nn.Upsample(scale_factor=2, mode='bilinear',#8x8->16x16
                                                align_corners=False),
                                ]),
                                ResConvBlock(c * 8, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#16x16->32x32
                            ]),
                            ResConvBlock(c * 8, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#32x32->64x64
                        ]),
                        ResConvBlock(c * 4, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#64x64->128x128
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#128x128->256x256
                ]),
                ResConvBlock(c * 2, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#256x256->512x512
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),# 512x512
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)







# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': args.diffusion_steps,
    'rescale_timesteps': True,
    'timestep_respacing': args.timesteps,   
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_checkpoint': False,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})
method = 'ddpm'  # ddim, ddpm





# Load models
import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model, diffusion = create_model_and_diffusion(**model_config)
image_size = 512

if image_size == 256:
    model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
else:
    model.load_state_dict(torch.load('512x512_diffusion_uncond_finetune_008100.pt', map_location='cpu'))

model.requires_grad_(False).eval().to(device)
if model_config['use_fp16']:
    model.convert_to_fp16()

if secondary_model_ver == 1:
    secondary_model = yyfc()
    secondary_model.load_state_dict(torch.load('yfcc_1.pth', map_location='cpu'),False)
    # secondary_model.load_state_dict(torch.load('wikiart_256.pkl', map_location='cpu'),False)
secondary_model.eval().requires_grad_(False).to(device)

clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)




def do_run():
    if seed is not None:
        torch.manual_seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
    side_x = side_y = model_config['image_size']

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / cutn] * cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            pred = secondary_model(x, cosine_t[None].repeat([n])).pred

            clip_in = normalize(make_cutouts(pred.add(1).div(2)))
            image_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([cutn, n, -1])
            clip_losses = dists.mul(weights).sum(2).mean(0)


            tv_losses = tv_loss(pred)
            range_losses = range_loss(pred)
            loss = clip_losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
            if init is not None and init_scale:
                init_losses = lpips_model(pred, init)
                loss = loss + init_losses.sum() * init_scale
            grad = -torch.autograd.grad(loss, x)[0]
            return grad

    if method == 'ddpm':
        sample_fn = diffusion.p_sample_loop_progressive
    elif method == 'ddim':
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        assert False




    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=True,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
        )


        for j, sample in enumerate(samples):
            if j % save_frequency == 0 or cur_t == 0:
                print()    
                for k, image in enumerate(sample['pred_xstart']):
                    filename = args.output + "_" + str(j) + "steps" + ".png"
                    if j>=500:
                        TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    display.display(display.Image(filename))
            cur_t -= 1

gc.collect()
do_run()