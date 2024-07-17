from latent_diffusion.backbone.transformer import Transformer
from guided_diffusion.patch_dataset import load_multipatchbag
from latent_diffusion.backbone.decoder import Reconstructor
from latent_diffusion.backbone.encoder import ImageEncoder

import torch.nn as nn
import torch
import argparse

def train(args: argparse.Namespace):
    patchbag = load_multipatchbag(
        wsi_dirs=args.wsi_dirs,
        h5_dirs=args.h5_dirs,
        batch_size=args.batch_size,
    )

    transformer = Transformer()
    encoder = ImageEncoder()
    model = Reconstructor(encoder=encoder, transformer=transformer)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):

        model.train()

        for i, patches in enumerate(patchbag):
            patches = patches.to(device)
            optimizer.zero_grad()

            outputs = model(patches)
            loss = criterion(outputs, patches)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(patchbag)}], Loss: {loss.item()}")
    
    torch.save(model.state_dict(), args.model_path)

def create_argparser():
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)

    # file params
    parser.add_argument("--wsi_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--h5_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    
    return parser

def main():
    parser = create_argparser()
    args = parser.parse_args()
    train(args)