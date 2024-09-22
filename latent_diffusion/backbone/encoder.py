# %%
from conch.open_clip_custom import create_model_from_pretrained
from torchvision import transforms
import torch.nn as nn
import torch

from dotenv import load_dotenv
from PIL import Image
import os

load_dotenv()
hf_auth_token = os.getenv("HF_AUTH_TOKEN")

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", 
            "hf_hub:MahmoodLab/conch", 
            hf_auth_token=hf_auth_token
        )

        self.trunk = model.visual.trunk # 784 tokens (28 * 28 patches as tokens)

        resize_target = 448

        self.pipe = transforms.Compose([
            transforms.Resize(size=resize_target, interpolation=Image.BICUBIC, antialias=True),
            transforms.CenterCrop(size=(resize_target,resize_target)),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # top left, top right, bottom left, bottom right
        
        x = self.pipe(x)
        with torch.no_grad():
            x = self.trunk(x)
        x = x[:, 1:, :] # remove CLS
        return x # B x 784 x 768

# %%
