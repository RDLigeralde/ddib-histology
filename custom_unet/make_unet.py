from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn
import torch

from typing import Tuple
import os

class UNet(nn.Module):
    def __init__(
        self,
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
        dropout: float = 0.0
    ):
        """
        HF UNet2DModel + util methods

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
        super().__init__()
        self.num_layers = len(layer_dims)
        self.img_size = img_size
        self.channels = channels
        self.time_embedding = time_embedding
        self.layer_dims = layer_dims
        self.layers_per_block = layers_per_block
        self.down_type = down_type
        self.up_type = up_type
        self.down_block_types = get_layer_types(default="DownBlock2D", alt="AttnDownBlock2D", alt_layer_idxs=attn_layers_down, num_layers=self.num_layers)
        self.up_block_types = get_layer_types(default="UpBlock2D", alt="AttnUpBlock2D", alt_layer_idxs=attn_layers_up, num_layers=self.num_layers)
        self.num_heads = num_heads
        self.dropout = dropout

        self.model = UNet2DModel(
            sample_size=self.img_size,
            in_channels=self.channels,
            out_channels=self.channels,
            time_embedding_type=self.time_embedding,
            block_out_channels=self.layer_dims,
            layers_per_block=self.layers_per_block,
            downsample_type=self.down_type,
            upsample_type=self.up_type,
            down_block_types=self.down_block_types,
            up_block_types=self.up_block_types,
            attention_head_dim=self.num_heads,
            dropout=self.dropout
        )
        self.dtype = self.model.dtype
        self.size_mb = sum(p.numel() for p in self.model.parameters()) * 4 / (1024 ** 2)

    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        return self.model(x, ts, return_dict=False)[0]
    
    def _make_arg_dict(self) -> dict[str, str]:
        return {
            "size_mb": str(self.size_mb),
            "img_size": str(self.img_size),
            "channels": str(self.channels),
            "time_embedding": self.time_embedding,
            "layer_dims": str(self.layer_dims),
            "layers_per_block": str(self.layers_per_block),
            "down_type": self.down_type,
            "up_type": self.up_type,
            "attn_layers_down": str(self.down_block_types),
            "attn_layers_up": str(self.up_block_types),
            "num_heads": str(self.num_heads),
            "dropout": str(self.dropout)
        }

    def write_arg_dict(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            arg_dict = self._make_arg_dict()
            for key, val in arg_dict.items():
                f.write(f"{key}: {val}\n")
    
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
    dropout: float = 0.0
) -> UNet:
    """
    Create a UNet

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
    return UNet(
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