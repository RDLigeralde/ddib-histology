'''
Calculates pairwise cosine similarity between 
CONCH embeddings of WSI patches from different
domains
'''
from latent_diffusion.backbone.encoder import ImageEncoder
from dataloading.patch_dataset import qpb_one_shot
from torch.utils.data import DataLoader
import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults_histology,
    classifier_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import argparse
import os

def pairwise_csim(x: torch.Tensor, y: torch.Tensor):
    '''
    Takes pairwise cosine similarities
    between B x H_CONCH x W_CONCH tensors
    '''
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1).T
    xn = x / torch.norm(x, dim=1, keepdim=True)
    ynt = y / torch.norm(y, dim=1, keepdim=True)
    return torch.mm(xn, ynt)

def translate(
    d1_batch, 
    model, 
    diffusion, 
    d1_weight_path, 
    d2_weight_path,
    in_channels,
    image_size,
    eta,
    device="cuda"
):
    # load source model
    model.load_state_dict(
        torch.load(d1_weight_path, map_location=device)
    )
    model.eval()
    model = model.to(device)

    d1_batch.to(device)
    d1_batch.to(device)

    # encoding
    noise_interm = diffusion.ddim_reverse_sample_loop_progressive(
        model,
        d1_batch,
        clip_denoised=False,
        model_kwargs=None,
        device=device
    )

    # load target model
    model.load_state_dict(
        torch.load(d2_weight_path, map_location=device)
    )
    sample_interm = diffusion.ddim_sample_loop_progressive(
        model,
        (d1_batch.shape[0], in_channels, image_size, image_size),
        noise=noise_interm[-1],
        clip_denoised=False,
        device=device,
        eta=eta,
    )
    return sample_interm[-1]

def csim_compare(
    d1_batch,
    d2_batch,
    model,
    diffusion,
    d1_weight_path,
    d2_weight_path,
    in_channels=3,
    image_size=256,
    eta=1 # TODO: find good default
):
    d1_ssim = pairwise_csim(d1_batch, d1_batch)
    d2_ssim = pairwise_csim(d2_batch, d2_batch)
    d1_d2_csim = pairwise_csim(d1_batch, d2_batch)
    print(f"Mean domain 1 cosine similarity: {torch.mean(d1_ssim)}")
    print(f"Mean domain 2 cosine similarity: {torch.mean(d2_ssim)}")
    print(f"Mean domain 1 <-> domain 2 cosine similarity: {torch.mean(d1_d2_csim)}")

    print("Translating domain 1 to domain 2...")
    d2_trans = translate(
        d1_batch,
        model,
        diffusion,
        d1_weight_path,
        in_channels,
        image_size,
        eta
    )
    d2_trans_csim = pairwise_csim(d2_batch, d2_trans)
    print(f"Mean domain 2 <-> domain 2 translated cosine similarity: {torch.mean(d2_trans_csim)}")

    print("Translating domain 2 to domain 1...")
    d1_trans = translate(
        d2_batch,
        model,
        diffusion,
        d2_weight_path,
        in_channels,
        image_size,
        eta
    )
    d1_trans_csim = pairwise_csim(d1_batch, d1_trans)
    print(f"Mean domain 1 <-> domain 1 translated cosine similarity: {torch.mean(d1_trans_csim)}")

def create_argparser():
    defaults = dict()
    defaults.update(model_and_diffusion_defaults_histology())
    defaults.update(classifier_defaults())
    
    parser = argparse.ArgumentParser()
    # Translation Defaults
    parser.add_argument("--image_out",type=str,default="",help="Path to save translated images")
    parser.add_argument("--eta",type=float,default=0.0,help="Diffusion eta")
    parser.add_argument("--resume_checkpoint",type=str,default="", help="Training resume point")
    parser.add_argument("--log_interval",type=int,default=100, help="Log interval")
    parser.add_argument("--save_interval",type=int,default=1000, help="Save interval")

    # Sampling Defaults
    parser.add_argument("--clip_denoised",type=bool,default=True)
    parser.add_argument("--num_samples",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--use_ddim",type=bool,default=True)
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--experiment",type=str,default=None)

    # Training Defaults
    parser.add_argument("--log_dir",type=str,default="./models/")
    parser.add_argument("--lr",type=float,default=1e-5, help="Learning rate")
    parser.add_argument("--microbatch",type=int,default=-1, help="Microbatch size")
    parser.add_argument("--ema_rate",type=float,default=0.9999, help="EMA rate")
    parser.add_argument("--schedule_sampler",type=str,default="uniform", help="Schedule sampler")
    parser.add_argument("--fp16_scale_growth",type=float,default=1e-3, help="FP16 scale growth")
    parser.add_argument("--weight_decay",type=float,default=0.0, help="Weight decay")
    parser.add_argument("--lr_anneal_steps",type=int,default=0, help="Learning rate anneal (decay) steps")
    parser.add_argument("--stop",type=int,default=10000, help="Max iteration count")

    # Model Params
    parser.add_argument("--image_size", type=int, help="Image size")
    parser.add_argument("--in_channels", type=int, help="Number of input channels")
    parser.add_argument("--num_channels", type=int, help="Number of input channels")
    parser.add_argument("--channel_mult", type=str, help="Channel multipliers") # WHY WOULD YOU MAKE THIS A STRING????
    parser.add_argument('--num_res_blocks', type=int, help="Number of residual blocks")
    parser.add_argument('--attention_resolutions', type=str, help="Attention resolutions") # WHY WOULD YOU MAKE THIS A STRING????
    parser.add_argument('--num_heads', type=int, help="Number of attention heads")

    # Diffusion Params
    parser.add_argument("--noise_schedule", type=str, help="Noise schedule")
    parser.add_argument("--diffusion_steps", type=int, help="Diffusion steps")

    # Transfer Params
    parser.add_argument("--transfer_method", type=str)

    # File Stuff
    parser.add_argument("--wsi_dir_d1",type=str,default="",help="Path to domain 1 WSIs")
    parser.add_argument("--coord_dir_d1",type=str,default="",help="Path to domain 1 H5s")
    parser.add_argument("--wsi_dir_d2",type=str,default="",help="Path to domain 2 WSIs")
    parser.add_argument("--coord_dir_d2",type=str,default="",help="Path to domain 2 H5s")
    parser.add_argument("--d1_weight_path",type=str,default="",help="Path to domain 1 state dict")
    parser.add_argument("--d2_weight_path",type=str,default="",help="Path to domain 2 state dict")

    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = create_argparser()
    args = parser.parse_args()

    print('Obtaining patch batches...')
    data_d1 = qpb_one_shot(
        wsi_dir=args.wsi_dir_d1,
        coord_dir=args.h5_dir_d1,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    data_d2 = qpb_one_shot(
        wsi_dir=args.wsi_dir_d2,
        coord_dir=args.h5_dir_d2,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    d1_batch, d2_batch = next(data_d1), next(data_d2)

    print('Creating model and diffusion...')
    arg_dict = args_to_dict(args, model_and_diffusion_defaults_histology().keys())
    model, diffusion = create_model_and_diffusion(
        **arg_dict
    )
    model = model.to(device)

    print('Calculating CONCH embeddings...')
    conch = ImageEncoder()
    d1_batch, d2_batch = conch(d1_batch), conch(d2_batch)

    csim_compare(
        d1_batch,
        d2_batch,
        model,
        diffusion,
        args.d1_weight_path,
        args.d2_weight_path,
        args.in_channels,
        args.image_size,
        args.eta
    )
    




    

    




