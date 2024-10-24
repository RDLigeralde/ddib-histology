from dataloading.patch_dataset import PatchDataset, tvt_split
from diffusers import SchedulerMixin, DDPMScheduler
from torch.utils.data import DataLoader
from custom_unet.make_unet import UNet
from torchvision import transforms
from torch.amp import GradScaler
import torch.optim as optim
import torch.nn as nn
import torch

from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm
import argparse
import os

def make_dls(
    wsi_dir: str, 
    coord_dir: str, 
    patch_size: int = 256, 
    patch_level: int = 0,
    num_timesteps: int = 1000,
    transform: transforms.Compose = None,
    scheduler: SchedulerMixin = DDPMScheduler,
    beta_schedule: str = "linear",
    predict_type: str = "noise",
    batch_size: int = 32,
    split: Tuple[int] = (.8, .1, .1),
    num_workers: int = 4,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders.

    Args:
        wsi_dir (str): WSI directory
        coord_dir (str): patch coordinate directory
        patch_size (int, optional): patch size (default: 256)
        patch_level (int, optional): patch level (default: 0)
        num_timesteps (int, optional): number of diffusion timesteps (default: 1000)
        transform (transforms.Compose, optional): image transformations (default: None)
        scheduler (SchedulerMixin, optional): diffusion scheduler (default: DDPMScheduler)
        beta_schedule (str, optional): beta schedule type (default: "linear")
        predict_type (str, optional): predict noise or image (default: "noise")
        batch_size (int, optional): batch size (default: 32)
        tvt_split (Tuple[int], optional): train / val / test split (default: (.8, .1, .1))
        num_workers (int, optional): number of workers (default: 4)
        persistent_workers (bool, optional): use persistent workers (default: False)
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: _description_
    """
    dataset = PatchDataset(
        wsi_dir=wsi_dir,
        coord_dir=coord_dir,
        patch_size=patch_size,
        patch_level=patch_level,
        transform=transform,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        beta_schedule=beta_schedule,
        predict_type=predict_type
    )
    train_set, val_set, test_set = tvt_split(dataset, split=split)
    train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    return train_dl, val_dl, test_dl

def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    load_path: str = None,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer: optim.Optimizer = optim.AdamW,
    out_dir: str = "models/histo_bright/model_00",
    devices: List[int] = [0],
    use_fp16: bool = False,
    save_interval: int = 1
):
    """
    Train a UNet model

    Args:
        model (nn.Module): _description_
        train_dl (DataLoader): _description_
        val_dl (DataLoader): _description_
        load_path (str, optional): _description_. Defaults to None.
        num_epochs (int, optional): _description_. Defaults to 10.
        lr (float, optional): _description_. Defaults to 1e-3.
        beta_schedule (str, optional): _description_. Defaults to "linear".
        out_dir (str, optional): _description_. Defaults to "models/histo_bright".
        devices (List[int], optional): GPUs to use (default: [0])
        save_interval (int, optional): save interval (default: 1)
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    hist_file = os.path.join(out_dir, "train_hist.log")
    err_file = os.path.join(out_dir, "train_err.log")
    param_log = os.path.join(out_dir, "model_params.txt")
    tp_log = os.path.join(out_dir, "train_params.txt")

    og_dataset = train_dl.dataset.dataset # dataset from subset
    with open(tp_log, "w") as tpf:
        tpf.write(f"Learning Rate: {lr}\n")
        tpf.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        tpf.write(f"Epochs: {epochs}\n")
        tpf.write(f"Mixed Precision: {'Enabled' if use_fp16 else 'Disabled'}\n")
        tpf.write(f"Num GPUs: {len(devices)}\n")
        tpf.write(f"Batch Size: {train_dl.batch_size}\n")
        tpf.write(f"Patch Size: {og_dataset.patch_size}\n")
        tpf.write(f"Patch Level: {og_dataset.patch_level}\n")
        tpf.write(f"Num Timesteps: {og_dataset.scheduler.config.num_train_timesteps}\n")
        tpf.write(f"Beta Schedule: {og_dataset.scheduler.config.beta_schedule}\n")
        tpf.write(f"Predict Type: {og_dataset.predict_type}\n")

    model.write_arg_dict(param_log)

    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    
    device = torch.device(f"cuda:{devices[0]}")
    scaler = GradScaler('cuda') if use_fp16 else None
    model = model.to(device)
    if load_path:
        model.load_state_dict(torch.load(load_path))
    if len(devices) > 1:
        model = nn.DataParallel(model, device_ids=devices)

    optimizer = optimizer(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_hist, val_hist = [], []

    hist = open(hist_file, "w")
    err = open(err_file, "w")
    
    with redirect_stdout(hist), redirect_stderr(err):

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            tl_tqdm = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", file=hist)
            for batch in tl_tqdm:
                optimizer.zero_grad()
                noisy_batch, target_batch, ts = batch
                noisy_batch, target_batch, ts = noisy_batch.to(device), target_batch.to(device), ts.to(device)
                if use_fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        pred_batch = model(noisy_batch, ts.squeeze(1))
                        loss = criterion(pred_batch, target_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_batch = model(noisy_batch, ts.squeeze(1))
                    loss = criterion(pred_batch, target_batch)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                tl_tqdm.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
                
            model.eval()
            with torch.no_grad():
                val_loss = 0
                vl_tqdm = tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
                for batch in vl_tqdm:
                    noisy_batch, target_batch, ts = batch
                    noisy_batch, target_batch, ts = noisy_batch.to(device), target_batch.to(device), ts.to(device)
                    pred_batch = model(noisy_batch, ts.squeeze(1))
                    loss = criterion(pred_batch, target_batch)
                    val_loss += loss.item()
                    vl_tqdm.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

                avg_train_loss = train_loss / len(train_dl)
                avg_val_loss = val_loss / len(val_dl)
                train_hist.append(avg_train_loss)
                val_hist.append(avg_val_loss)

            if epoch % save_interval == 0:
                torch.save(model.module.state_dict(), os.path.join(out_dir, f"model_{epoch:03d}.pt"))
    
    plot_hist(train_hist, val_hist, out_dir)

def plot_hist(train_hist, val_hist, out_dir):
    plt.plot(train_hist, label="Train Loss", color="blue")
    plt.plot(val_hist, label="Val Loss", color="orange")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(len(train_hist)))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, "train_hist.png"))
    plt.close()

def create_argparser():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--wsi_dir", type=str, required=True, help="WSI directory")
    parser.add_argument("--coord_dir", type=str, required=True, help="patch coordinate directory")
    parser.add_argument("--out_dir", type=str, default="models/histo_bright/model_00", help="output directory")

    # Dataset
    parser.add_argument("--patch_size", type=int, default=256, help="patch size")
    parser.add_argument("--patch_level", type=int, default=0, help="patch level")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="beta schedule type")
    parser.add_argument("--predict_type", type=str, default="noise", help="predict noise or image")

    # Model
    parser.add_argument("--channels", type=int, default=3, help="number of input channels")
    parser.add_argument("--time_embedding", type=str, default="positional", help="time embedding type")
    parser.add_argument("--layer_dims", type=int, nargs="+", default=[256, 256, 512, 1024], help="layer dimensions")
    parser.add_argument("--layers_per_block", type=int, default=2, help="layers per block")
    parser.add_argument("--down_type", type=str, default="ResNet", help="downsampling type")
    parser.add_argument("--up_type", type=str, default="ResNet", help="upsampling type")
    parser.add_argument("--attn_layers_down", type=int, nargs="+", default=[2,3], help="downsampling attention layers")
    parser.add_argument("--attn_layers_up", type=int, nargs="+", default=[0,1], help="upsampling attention layers")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="GPU devices")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--persistent_workers", action="store_true", help="use persistent workers")
    parser.add_argument("--use_fp16", action="store_true", help="use mixed precision training")
    parser.add_argument("--save_interval", type=int, default=1, help="save interval")

    return parser

def main():
    parser = create_argparser()
    args = parser.parse_args()
    train_dl, val_dl, _ = make_dls(
        args.wsi_dir,
        args.coord_dir,
        args.patch_size,
        args.patch_level,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        predict_type=args.predict_type,
        batch_size=args.batch_size,
    )
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
    train(
        model,
        train_dl,
        val_dl,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        devices=args.devices,
        use_fp16=args.use_fp16
    )

if __name__ == "__main__":
    main()

    

    


        

        