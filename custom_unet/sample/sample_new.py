from custom_unet.make_unet import UNet
import torch.nn as nn
import numpy as np
import torch

import matplotlib.pyplot as plt
from typing import List, Union
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
    timesteps: List[int],
    sampling_method: str,
    beta_schedule: str = "linear",
    device: str = "cuda",
    num_train_timesteps: int = 1000,
    etas: List[float] = [0.0]  # only relevant for DDIM
) -> List[np.ndarray]:
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

    for t in timesteps: # only one of these loops matters in practice
        for eta in etas:
            images = pipe(
                batch_size=batch_size,
                num_inference_steps=t,
                eta=eta,
                output_type='np.array'
            )['images']
            all_images.append(images)

    return all_images
    
def plot_images(images: List[np.ndarray], out_dir: str, values: List, const: int, experiment: str):
    num_rows = len(images)
    num_cols = images[0].shape[0]
    
    _, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    if num_rows == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]
    
    for i, img_batch in enumerate(images):
        for j, img in enumerate(img_batch):
            axs[i, j].imshow(img)
        
        axs[i, 0].set_ylabel(f"x={values[i]}", fontsize=16)
    
    for j in range(num_cols):
        axs[0, j].set_title(f"Image {j + 1}", fontsize=16)
    
    plt.tight_layout()
    fixed = "timesteps" if experiment == "eta" else "eta"
    plt.title(f"Sampling Test: {experiment} ({fixed}=={const})")
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
    parser.add_argument("--experiment", type=str, help="eta or timestep")
    parser.add_argument("--sampling_method", type=str, help="ddim or ddpm")
    parser.add_argument("--min", type=float)
    parser.add_argument("--max", type=float)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--eta_fixed", type=float, default=0.0)
    parser.add_argument("--timesteps_fixed", type=int, default=100)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--device", type=str, default="cuda")
    
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
    parser.add_argument("--num_train_timesteps", type=int, default=1000)

    args = parser.parse_args()
    return args

def main():
    args = create_argparser()
    model_dir = os.path.dirname(args.model_path)

    if args.experiment == 'eta':
        etas = np.linspace(
            args.min,
            args.max,
            args.count,
            dtype=np.float32
        )
        timesteps = [args.timesteps_fixed]
        plt_path = os.path.join(model_dir, f"{args.batch_size}_samples_{args.sampling_method}_etarange_{args.min}_{args.max}_tsteps_{args.timesteps_fixed}.png")
        values, const = etas, timesteps[0]
    else:
        timesteps = np.linspace(
            args.min,
            args.max,
            args.count,
            dtype=np.int32
        )
        etas = [args.eta_fixed]
        plt_path = os.path.join(model_dir, f"{args.batch_size}_samples_{args.sampling_method}_tsteprange_{args.min}_{args.max}_eta_{args.eta_fixed}.png")
        values, const = timesteps, etas[0]
        
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

    

    images = sample_images(
        model, 
        args.batch_size, 
        timesteps, 
        args.sampling_method, 
        args.beta_schedule,
        args.device,
        args.num_train_timesteps,
        etas
    )
    plot_images(images, plt_path, values, const, args.experiment)

if __name__ == "__main__":
    main()