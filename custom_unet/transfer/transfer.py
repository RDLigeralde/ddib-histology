from dataloading.patch_dataset import PatchDataset
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
from custom_unet.make_unet import UNet
from torchvision import transforms
import torch.nn as nn
import torch

from custom_unet.transfer.reverse_sample_utils import (
    diffusion_encode, 
    diffusion_decode, 
    plot_transfer
)
from typing import List
import argparse
import os

def transfer_images(
    source_model: nn.Module,
    target_model: nn.Module,
    source_wsi_dir: str,
    source_coord_dir: str,
    num_images: int,
    scheduler_type,
    patch_size: int = 256,
    patch_level: int = 0,
    predict_type: str = "noise",
    transforms: transforms.Compose = None,
    num_timesteps: int = 1000,
    num_inference_timesteps: int = 1000,
    beta_schedule: str = "linear",
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Complete style transfer loop"""
    pd = PatchDataset(
        wsi_dir=source_wsi_dir,
        coord_dir=source_coord_dir,
        patch_size=patch_size,
        patch_level=patch_level,
        transform=transforms,
        num_timesteps=num_timesteps,
        scheduler=DDIMScheduler,
        beta_schedule=beta_schedule,
        predict_type=predict_type,
        eval=True
    )
    pdl = DataLoader(pd, batch_size=num_images, shuffle=True)
    source_model.to(device)
    target_model.to(device)
    source_model.eval()
    target_model.eval()

    if scheduler_type == DDPMScheduler and predict_type == "epsilon":
        predict_type = "noise"
        scheduler = scheduler_type(num_train_timesteps=num_timesteps, beta_schedule=beta_schedule, predict_type=predict_type)
    else:
        scheduler = scheduler_type(num_train_timesteps=num_timesteps, beta_schedule=beta_schedule, prediction_type=predict_type)

    scheduler.set_timesteps(num_inference_timesteps)

    with torch.inference_mode():
        _, original, _, _ = next(iter(pdl))
        original = original.to(device)
        
        encode_inters = diffusion_encode(
            original, 
            source_model, 
            scheduler, 
            num_inference_timesteps,
            device
        )
        
        decode_inters = diffusion_decode(
            encode_inters[-1],
            target_model,
            scheduler,
            num_inference_timesteps,
            device
        )

    return encode_inters, decode_inters

def create_argparser():
    NOISERS = {
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler
    }
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", type=str)

    parser.add_argument("--source_model_path", type=str)
    parser.add_argument("--target_model_path", type=str)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="positional")
    parser.add_argument("--layer_dims", type=int, nargs="+", default=[256, 512, 768])
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--down_type", type=str, default="conv")
    parser.add_argument("--up_type", type=str, default="conv")
    parser.add_argument("--attn_layers_down", type=int, nargs="+", default=[2])
    parser.add_argument("--attn_layers_up", type=int, nargs="+", default=[0])
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--source_wsi_dir", type=str)
    parser.add_argument("--source_coord_dir", type=str)
    parser.add_argument("--num_images", type=int)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=NOISERS.keys())
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--predict_type", type=str, default="epsilon")

    args = parser.parse_args()
    args.scheduler_type = NOISERS[args.scheduler_type]
    return args

def _deparallelize_state_dict(state_dict: dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def main():
    args = create_argparser()
    print('Creating source and target models...')
    source_model = UNet(
        args.patch_size,
        args.channels,
        args.time_embedding,
        args.layer_dims,
        args.layers_per_block,
        args.down_type,
        args.up_type,
        args.attn_layers_down,
        args.attn_layers_up,
        args.num_heads,
        args.dropout
    )
    target_model = UNet(
        args.patch_size,
        args.channels,
        args.time_embedding,
        args.layer_dims,
        args.layers_per_block,
        args.down_type,
        args.up_type,
        args.attn_layers_down,
        args.attn_layers_up,
        args.num_heads,
        args.dropout
    )
    print('Loading source and target models...')
    source_sd = torch.load(args.source_model_path, map_location=args.device, weights_only=False)
    target_sd = torch.load(args.target_model_path, map_location=args.device, weights_only=False)
    if True: # HACK: Temporary fix for DataParallel
        source_sd = _deparallelize_state_dict(source_sd)
        target_sd = _deparallelize_state_dict(target_sd)
    source_model.to(args.device)
    target_model.to(args.device)
    source_model.load_state_dict(source_sd)
    target_model.load_state_dict(target_sd)

    model_dir = os.path.dirname(args.source_model_path)
    scheduler_name = args.scheduler_type.__name__.lower()[:4]
    plt_path = os.path.join(model_dir, f"{args.num_images}_transfer_{scheduler_name}_{args.num_inference_timesteps}.png")

    print('Transferring images...')
    encode_inters, decode_inters = transfer_images(
        source_model, 
        target_model,
        args.source_wsi_dir,
        args.source_coord_dir,
        args.num_images,
        args.scheduler_type,
        args.patch_size,
        args.patch_level,
        args.predict_type,
        num_timesteps=args.num_timesteps,
        num_inference_timesteps=args.num_inference_timesteps,
        beta_schedule=args.beta_schedule,
        device=args.device
    )
    print('Plotting transfer...')
    plot_transfer(encode_inters, decode_inters, args.title, save_path=plt_path)

if __name__ == "__main__":
    main()