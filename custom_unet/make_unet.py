from diffusers import UNet2DModel
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn
import torch

from typing import Tuple

class UNetLightning(pl.LightningModule):
    def __init__(self, UNet2dModel: UNet2DModel, lr: float = 1e-3):
        """
        Pytorch Lightning Wrapper for UNet2DModel

        Args:
            UNet2dModel (UNet2DModel): model
            lr (float, optional): learning rate (default: 1e-3)
        """
        super().__init__()
        self.model = UNet2dModel
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
def get_layer_types(default: str, alt: str, alt_layer_idxs: Tuple[int], num_layers: int) -> Tuple[str]:
    """Return layer type list"""
    layer_types = [default] * num_layers
    for idx in alt_layer_idxs:
        layer_types[idx] = alt
    return tuple(layer_types)
    
def make_unet(
    img_size: int,
    channels: int,
    time_embedding: str,
    layer_dims: Tuple[int],
    layers_per_block: int,
    down_type: str,
    up_type: str,
    attn_layers_down: Tuple[int],
    attn_layers_up: Tuple[int],
    num_heads: int,
    dropout: float = 0.0,
) -> UNet2DModel:
    """
    Create a UNet2DModel

    Args:
        img_size (int): patch H, W
        channels (int): color channels
        time_embedding (str): time embedding type
        layer_dims (Tuple[int]): output dim of each layer
        layers_per_block (int): number of layers per block
        down_type (str): downsampling layer type (convolutional or ResNet)
        up_type (str): upsampling layer type (convolutional or ResNet)
        attn_layers_down (Tuple[int]): downsampling attention layer indices
        attn_layers_up (Tuple[int]): upsampling attention layer indices
        num_heads (int): number of attention heads
        dropout (float, optional): dropout rate (default: 0.0)
    """
    num_layers = len(layer_dims)
    down_block_types = get_layer_types(default="DownBlock2D", alt="AttnDownBlock2D", alt_layer_idxs=attn_layers_down, num_layers=num_layers)
    up_block_types = get_layer_types(default="UpBlock2D", alt="AttnUpBlock2D", alt_layer_idxs=attn_layers_up, num_layers=num_layers)
    return UNet2DModel(
        sample_size=img_size,
        in_channels=channels,
        out_channels=channels,
        time_embedding=time_embedding,
        layer_dims=layer_dims,
        layers_per_block=layers_per_block,
        downsample_type=down_type,
        upsample_type=up_type,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        num_heads=num_heads,
        dropout=dropout
    )


def make_unet_pl( 
    img_size: int,
    channels: int,
    time_embedding: str,
    layer_dims: Tuple[int],
    layers_per_block: int,
    down_type: str,
    up_type: str,
    attn_layers_down: Tuple[int],
    attn_layers_up: Tuple[int],
    num_heads: int,
    dropout: float = 0.0,
    lr: float = 1e-3,
) -> UNetLightning:
    """
    Create a UNetLightning model

    Args:
        img_size (int): patch H, W
        channels (int): color channels
        time_embedding (str): time embedding type
        layer_dims (Tuple[int]): output dim of each layer
        down_type (str): downsampling layer type (convolutional or ResNet)
        up_type (str): upsampling layer type (convolutional or ResNet)
        attn_layers_down (Tuple[int]): downsampling attention layer indices
        attn_layers_up (Tuple[int]): upsampling attention layer indices
        num_heads (int): number of attention heads
        dropout (float, optional): dropout rate (default: 0.0)
    """
    model = make_unet(
        img_size, 
        channels, 
        time_embedding, 
        layer_dims,
        layers_per_block,
        down_type, 
        up_type, 
        attn_layers_down, 
        attn_layers_up, 
        num_heads, 
        dropout
    )
    return UNetLightning(model, lr)