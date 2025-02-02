# %%
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt


@torch.inference_mode()
def diffusion_encode(
    batch: torch.Tensor,
    model: nn.Module,
    scheduler: Union[DDIMScheduler, DDPMScheduler],
    num_timesteps: int,
    device: Union[str, torch.device] = "cuda"
) -> List[torch.Tensor]:
    """
    Uses noise predictions to run diffusion sampling backwards
    Based on: https://huggingface.co/learn/diffusion-course/en/unit4/2

    Args:
        batch (torch.Tensor): batch of images from source domain
        model (nn.Module): model trained on source domain
        scheduler (Union[DDIMScheduler, DDPMScheduler]): scheduler used to generate noise predictions
        num_timesteps (int): number of timesteps to run
        device (Union[str, torch.device], optional): device to encode on (default: "cuda")

    Returns:
        List[torch.Tensor]: _description_
    """
    
    batch = batch.to(device)
    model.to(device)
    timesteps = reversed(scheduler.timesteps)
    latents = []
    latents.append(batch)

    for i in range(1, num_timesteps):
        t = timesteps[i]
        noise_pred = model(batch, t)
        t_prev = max(1, t.item() - (1000 // num_timesteps))
        alpha_t = scheduler.alphas_cumprod[t.item()]
        alpha_t_next = scheduler.alphas_cumprod[t_prev]

        if isinstance(scheduler, DDPMScheduler): # Reverse DDPM update rule
            batch = alpha_t_next * noise_pred + (1 - alpha_t_next) * batch
        else: # Reverse DDIM update rule
            batch = (batch - (1-alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1-alpha_t_next).sqrt() * noise_pred
        latents.append(batch)

    return latents

@torch.inference_mode()
def diffusion_decode(
    batch: torch.Tensor,
    model: nn.Module,
    scheduler: Union[DDIMScheduler, DDPMScheduler],
    num_timesteps: int,
    device: Union[str, torch.device] = "cuda"
):
    """
    Takes latents from diffusion_encode and runs forward sampling
    """
    batch = batch.to(device)
    model.to(device)
    scheduler.set_timesteps(num_timesteps)
    timesteps = scheduler.timesteps
    preds_curr = batch
    preds = [preds_curr]

    for i in range(1, num_timesteps):
        t = timesteps[i]
        noise_pred = model(preds_curr, t)
        preds_curr = scheduler.step(model_output=noise_pred, timestep=t, sample=batch).prev_sample
        preds.append(preds_curr)
    
    return preds

def plot_transfer(
    latents_encode: List[torch.Tensor], 
    latents_decode: List[torch.Tensor],
    title: str,
    save_path: Optional[str] = None,
    save_interval: int = 10
):
    steps = list(range(0, len(latents_encode), save_interval))
    n_steps = len(steps)
    num_timesteps = len(latents_encode)
    batch_size = latents_encode[0].shape[0]
    num_rows, num_cols = 2 * num_timesteps, batch_size
    fig, ax = plt.subplots(2 * num_timesteps, batch_size, figsize=(num_cols * 5, num_rows * 5))
    
    for i, t in enumerate(steps):
        for j in range(batch_size):
            encode_i = latents_encode[t][j].cpu().detach().numpy().transpose(1, 2, 0)
            encode_i = (encode_i - encode_i.min()) / (encode_i.max() - encode_i.min())
            ax[i, j].imshow(encode_i)
            ax[i, j].axis("off")
            
            if i == 0:
                ax[i, j].set_title(f"Image {j}")
            
            ax[i, j].set_ylabel(f"t={t}", rotation=0, labelpad=40)
    
    for i, t in enumerate(steps):
        for j in range(batch_size):
            decode_i = latents_decode[t][j].cpu().detach().numpy().transpose(1, 2, 0)
            decode_i = (decode_i - decode_i.min()) / (decode_i.max() - decode_i.min())
            ax[i + n_steps, j].imshow(decode_i)
            ax[i + n_steps, j].axis("off")
            ax[i + n_steps, j].set_ylabel(f"t={n_steps-t}", rotation=0, labelpad=40)
    
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

    



        



    


# %%
