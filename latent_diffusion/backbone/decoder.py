# %%
from latent_diffusion.backbone.positional_encoding import PositionalEncoding
from latent_diffusion.backbone.transformer import Transformer
import torch.nn as nn
import torch

class ImageDecoder(nn.Module):
    def __init__(
            self, 
            transformer: Transformer, 
            img_size: int = 448,
            num_channels: int = 3,
            embed_dim: int = 768,
            num_tokens: int = 784,
            
        ):
        """
        Reconstructs images from CONCH embeddings

        Args:
            transformer (Transformer): transformer model
            img_size (int, optional): target image size
            embed_dim (int, optional): latent space dimension
            num_tokens (int, optional): number of tokens
            num_channels (int, optional): number of color channels
        """
        super().__init__()
        self.num_channels = num_channels
        self.img_size = img_size

        self.pe = PositionalEncoding(num_tokens, embed_dim)
        self.transformer = transformer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        embed_current = x
        embed_initial = embed_current.clone()  # or another embedding if available
        embed_current_pe = self.pe(embed_current)

        transformed = self.transformer(embed_current, embed_initial, embed_current_pe)
        output = self.output_layer(transformed)
        output = output.view(batch_size, self.num_channels, self.img_size, self.img_size)

        return output

# %%
