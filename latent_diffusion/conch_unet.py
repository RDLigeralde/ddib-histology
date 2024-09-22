from torch import Tensor
import torch.nn as nn
import torch

from typing import List, Tuple, Optional

from guided_diffusion.script_util import create_gaussian_diffusion
from latent_diffusion.backbone.encoder import ImageEncoder
from guided_diffusion.unet import UNetModel

class ConchNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        unet_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: List[int],
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        dropout: float = 0.0,
        channel_mult: Tuple[int] = (1,2,4,8),
        conv_resample: bool = True,
        dims: int = 3,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = True
    ):
        """
        Latent Diffusion UNet with CONCH encoder backbone

        Args:
            in_channels (int): number of CONCH output channels
            unet_channels (int): number of UNet input channels
            out_channels (int): number of output channels
            model_channels (int): UNet model channels
            num_res_blocks (int): number of residual blocks
            attention_resolutions (List[int]): layers to perform self-attention
            num_heads (int, optional): number of attention heads (default: 1)
            num_head_channels (int, optional): number of attention head channels (default: -1)
            num_heads_upsample (int, optional): number of attention head channels for upsampling (default: -1)
            dropout (float, optional): dropout rate (default: 0.0)
            channel_mult (Tuple[int], optional): channel multiplier at layer i. (default: (1,2,4,8))
            conv_resample (bool, optional): use convolutional resampling (default: True)
            dims (int, optional): 1D, 2D, or 3D input (default: 3)
            num_classes (Optional[int], optional): number of classes for conditioning (default: None)
            use_checkpoint (bool, optional): use gradient checkpointing (default: False)
            use_fp16 (bool, optional): use 16-bit precision (default: False)
            use_scale_shift_norm (bool, optional): TODO: figure out what this does (default: False)
            resblock_updown (bool, optional): use residual blocks for upsampling (default: False)
        """
        super().__init__()

        self.encoder = ImageEncoder()
        self.project = nn.Conv2d(in_channels, unet_channels, 1)
        self.UNet = UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.project(x)
        return self.UNet(x)
    
def create_conchnet(
    in_channels: int,
    unet_channels: int,
    out_channels: int,
    model_channels: int,
    num_res_blocks: int,
    attention_resolutions: List[int],
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    dropout: float = 0.0,
    channel_mult: Tuple[int] = (1,2,4,8),
    conv_resample: bool = True,
    dims: int = 3,
    num_classes: Optional[int] = None,
    use_checkpoint: bool = False,
    use_fp16: bool = False,
    use_scale_shift_norm: bool = False,
    resblock_updown: bool = True,
    learn_sigma: bool = False
) -> ConchNet:
    """
    Create a ConchNet model. See class for details
    """

    if unet_channels == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif unet_channels == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif unet_channels == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif unet_channels == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {unet_channels}")

    attention_ds = []
    for attention_resolution in attention_resolutions:
        attention_ds.append(in_channels // int(attention_resolution))

    out_channels = in_channels if not learn_sigma else in_channels * 2 # TODO: is the else correct?
    return ConchNet(
        in_channels=in_channels,
        unet_channels=unet_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_ds,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        dims=dims,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown
    )

def create_conchnet_and_diffusion(
    in_channels: int,
    unet_channels: int,
    out_channels: int,
    model_channels: int,
    num_res_blocks: int,
    attention_resolutions: List[int],
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    dropout: float = 0.0,
    channel_mult: Tuple[int] = (1,2,4,8),
    conv_resample: bool = True,
    dims: int = 3,
    num_classes: Optional[int] = None,
    use_checkpoint: bool = False,
    use_fp16: bool = False,
    use_scale_shift_norm: bool = False,
    resblock_updown: bool = True,
    learn_sigma: bool = False,
    steps: int = 1000,
    noise_schedule: str = "Linear",
    use_kl: bool = False,
    predict_xstart: bool = False,
    rescale_timesteps: bool = False,
    rescale_learned_sigmas: bool = False,
    timestep_respacing: str = "",
):
    conchnet = create_conchnet(
        in_channels=in_channels,
        unet_channels=unet_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        dims=dims,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown
    )
    diffusion = create_gaussian_diffusion(
        steps=steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing
    )

    return conchnet, diffusion
    s








