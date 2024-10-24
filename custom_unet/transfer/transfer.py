from dataloading.patch_dataset import PatchDataset
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
from custom_unet.make_unet import UNet
from torchvision import transforms
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from typing import List
import argparse
import os

def transfer_images(
    target_model: nn.Module,
    source_wsi_dir: str,
    source_coord_dir: str,
    num_images: int,
    scheduler_type, # how to type hint this?
    patch_size: int = 256,
    patch_level: int = 0,
    predict_type: str = "noise",
    transforms: transforms.Compose = None,
    num_timesteps: int = 1000,
    num_inference_timesteps: int = 1000,
    beta_schedule: str = "linear",
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Transfer source domain images to target domain via trained diffusion model

    Args:
        target_model (nn.Module): diffusion model trained on target domain
        source_wsi_dir (str): directory containing WSIs from source domain
        source_h5_dir (str): directory containing coordinates from source domain
        num_images (int): number of images to transfer
        beta_schedule (str, optional): beta schedule (default: linear)
        device (str, optional): device to run model on (default: cuda)

    Returns:
        List[torch.Tensor]: list of batched transfer images every 100 timesteps
    """
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
    images = []
    target_model.to(device)
    target_model.eval()

    if scheduler_type == DDPMScheduler and predict_type == "epsilon":
        predict_type = "noise"
        scheduler = scheduler_type(num_train_timesteps=num_timesteps, beta_schedule=beta_schedule, predict_type=predict_type)
    else: # DDIM takes different kwarg
        scheduler = scheduler_type(num_train_timesteps=num_timesteps, beta_schedule=beta_schedule, prediction_type=predict_type)

    scheduler.set_timesteps(num_inference_timesteps)

    with torch.inference_mode():
        steps = scheduler.timesteps
        noised, original, _, _  = next(iter(pdl))
        noised = noised.to(device)
        images.append(original)
        for step in steps:
            noise_pred = target_model(noised, step)
            noised = scheduler.step(model_output=noise_pred, timestep=step, sample=noised).prev_sample
            if step % 100 == 0:
                images.append(noised.detach().cpu())

    return images

def plot_images(images: List[torch.Tensor], out_dir: str):
    num_rows = len(images)
    num_cols = images[0].shape[0]
    
    _, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]
    
    for i, img_batch in enumerate(images):
        for j, img in enumerate(img_batch):
            img = img.permute(1, 2, 0).clamp(0, 1)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
        
    plt.savefig(out_dir)
    plt.close()

def create_argparser():
    NOISERS = {
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler
    }
    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument("--model_path", type=str)
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

    # Data Args
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--source_wsi_dir", type=str)
    parser.add_argument("--source_coord_dir", type=str)
    parser.add_argument("--num_images", type=int)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    # Noising Args
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
    model = UNet(
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
    sd = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if True:  # HACK until I figure out how to get the --dataparallel flag to work
        sd = _deparallelize_state_dict(sd) 
    model.to(args.device)
    model.load_state_dict(sd)

    model_dir = os.path.dirname(args.model_path)
    scheduler_name = args.scheduler_type.__name__.lower()
    plt_path = os.path.join(model_dir, f"{args.num_images}_transfer_{scheduler_name}.png")

    images = transfer_images(
        model,
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
    plot_images(images, plt_path)

if __name__ == "__main__":
    main()



    
    