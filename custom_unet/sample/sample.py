from custom_unet.make_unet import UNet
from diffusers import DDPMScheduler
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from typing import List
import argparse
import os

def sample_images(
    model: nn.Module,
    num_images: int,
    num_timesteps: int,
    beta_schedule: str = "linear",
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Sample images from the model

    Args:
        model (nn.Module): model to sample from
        num_images (int): number of images to sample
        num_timesteps (int): number of timesteps to sample
        beta_schedule (str, optional): beta schedule (default: "linear")
        device (str, optional): device to sample on (default: "cuda")
    Returns:
        Tensor: sampled images
    """
    images = []
    model.eval()
    img_size, channels = model.img_size, model.channels
    noise_curr = torch.randn(num_images, channels, img_size, img_size).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, beta_schedule=beta_schedule)
    with torch.inference_mode():
        steps = scheduler.timesteps
        for step in steps:
            noise_pred = model(noise_curr, step)
            noise_curr = scheduler.step(model_output=noise_pred, timestep=step, sample=noise_curr).prev_sample

            if step % 100 == 0:
                images.append(noise_curr.detach().cpu())
    
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
        
        axs[i, 0].set_ylabel(f"Step {i * 100}", fontsize=16)
    
    for j in range(num_cols):
        axs[0, j].set_title(f"Image {j + 1}", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_dir)
    plt.show()

def _deparallelize_state_dict(state_dict: dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def create_argparser():
    parser = argparse.ArgumentParser()

    # Sample Args
    parser.add_argument("--num_images", type=int, help="Number of images to sample")
    parser.add_argument("--num_timesteps", type=int, help="Number of timesteps to sample")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule")
    parser.add_argument("--device", type=str, default="cuda", help="Device to sample on")

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

    args = parser.parse_args()
    return args

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
    if True: # HACK until I figure out how to get the --dataparallel flag to work
        sd = _deparallelize_state_dict(sd)
    model.to(args.device)
    model.load_state_dict(sd)

    model_dir = os.path.dirname(args.model_path)
    plt_path = os.path.join(model_dir, f"{args.num_images}_samples.png")

    images = sample_images(model, args.num_images, args.num_timesteps, args.beta_schedule, args.device)
    plot_images(images, plt_path)

if __name__ == "__main__":
    main()

