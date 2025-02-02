from custom_unet.make_unet import UNet
import torch.nn as nn
import numpy as np
import torch

import matplotlib.pyplot as plt
from typing import List
from PIL import Image
import argparse
import os

from diffusers import (
    DDIMPipeline, 
    DDPMPipeline,
    DDIMScheduler,
    DDPMScheduler
)

def sample_images(
    model: nn.Module,
    batch_size: int,
    timesteps: List[int],  # List of timesteps to sample at
    sampling_method: str,
    beta_schedule: str = "linear",
    device: str = "cuda",
    num_train_timesteps: int = 1000,
    eta: float = 0.0  # only relevant for DDIM
) -> List[List[Image.Image]]:
    model.eval()
    if sampling_method == "ddim":
        pipe_type = DDIMPipeline
        sched_type = DDIMScheduler
    else:
        pipe_type = DDPMPipeline
        sched_type = DDPMScheduler

    scheduler = sched_type(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule
    )
    if hasattr(scheduler, "set_timesteps"):
        scheduler.set_timesteps(num_train_timesteps)

    pipe = pipe_type(model, scheduler).to(device)

    all_images = []
    for t in timesteps:
        images = pipe(
            batch_size=batch_size,
            num_inference_steps=t,
            eta=eta
        )['images']
        all_images.append(images)

    return all_images
    
def plot_images(images: List[List[Image.Image]], out_dir: str, timesteps: List[int]):
    num_rows = len(images)  # Number of timesteps
    num_cols = len(images[0])  # Number of images per timestep

    _, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]

    for i, img_batch in enumerate(images):  # Iterate over timesteps
        for j, img in enumerate(img_batch):  # Iterate over images in the batch
            img_array = np.array(img)

            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0

            axs[i][j].imshow(img_array)
            axs[i][j].axis('off')

        axs[i][0].set_ylabel(f"Step {timesteps[i]}", fontsize=16)  # Label for the timestep

    for j in range(num_cols):
        axs[0][j].set_title(f"Image {j + 1}", fontsize=16)

    plt.tight_layout()
    plt.savefig(out_dir)
    plt.show()

def _deparallelize_state_dict(state_dict: dict): # because I'm bad at saving models
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def _reformat_state_dict(state_dict):
    new_state_dict = {}
    prefix = "model."
    
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    return new_state_dict

def create_argparser():
    parser = argparse.ArgumentParser()

    # Sample Args
    parser.add_argument("--batch_size", type=int, help="Number of images to sample")
    parser.add_argument("--timestep_min", type=int, default=100, help="Minimum timestep to sample at")
    parser.add_argument("--timestep_max", type=int, default=1000, help="Maximum timestep to sample at")
    parser.add_argument("--timestep_count", type=int, default=10, help="Number of timesteps to sample at")
    parser.add_argument("--eta", type=float)
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule")
    parser.add_argument("--device", type=str, default="cuda", help="Device to sample on")
    parser.add_argument("--sampling_method", type=str)

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
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of timesteps to sample")

    args = parser.parse_args()
    return args

def main():
    args = create_argparser()
    timesteps = np.linspace(
        args.timestep_min,
        args.timestep_max,
        args.timestep_count,
        dtype=int
    ).tolist()
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
    ).model
    sd = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if True: # HACK until I figure out how to get the --dataparallel flag to work
        sd = _deparallelize_state_dict(sd)
        sd = _reformat_state_dict(sd)
    model.to(args.device)
    model.load_state_dict(sd)

    model_dir = os.path.dirname(args.model_path)
    plt_path = os.path.join(model_dir, f"{args.batch_size}_samples_{args.sampling_method}_tsteps_{args.timestep_min}_{args.timestep_max}_{args.timestep_count}_eta_{args.eta}.png")

    images = sample_images(
        model, 
        args.batch_size, 
        timesteps, 
        args.sampling_method, 
        args.beta_schedule,
        args.device,
        args.num_train_timesteps,
        args.eta
    )
    plot_images(images, plt_path)

if __name__ == "__main__":
    main()