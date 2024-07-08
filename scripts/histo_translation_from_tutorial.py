import random
import os
import torch
from PIL import Image
import numpy as np
import time
import argparse
from argparse import Namespace
import torch.nn.functional as F

from guided_diffusion.script_util import model_and_diffusion_defaults, classifier_defaults
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_source_data_for_domain_translation, load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
)



def montage(grid_size, dataset_directory, dataloader=None, resize_scale=1.0, dpi=300):
    """Create image montage sampled from dataset"""
    # TO DO : CHOOSE SAME SAMPLES FOR DATASET AND DATALOADER 
    number_images = int(grid_size ** 2)
    filepaths = list()
    imgs = list()
    file_prefix = f"{dataset_directory}"
    
    if dataloader != None:
        counter = 0
        for batch in dataloader:
            for img in batch[0]:
                img = img.cpu().numpy().transpose((1, 2, 0))
                img = np.squeeze(img)
                img = np.uint8((img + 1.) * 255.)
                print(img)
                print(img.shape)
                img = Image.fromarray(img)
                resizing_factor = 1 / grid_size * resize_scale
                if resizing_factor < 1.0: # Ensures that images are not upsampled
                    new_size = tuple(int(dim * resizing_factor) for dim in img.size)
                    img = img.resize(new_size)
                imgs.append(img)
                counter += 1
                if counter + 1 == number_images:
                    break
        original_size = imgs[0].size
        file_prefix += "_dataloader"

    else:
        # Get filepaths
        for root, subdirs, files in os.walk(dataset_directory):
            for f in files:
                if f.endswith("png"):
                    filepaths.append(os.path.join(root, f))
        assert number_images <= len(filepaths), f"too few images in directory for montage"
        original_size = Image.open(filepaths[0]).size
        
        # Sample filepaths and load images
        filepaths = random.sample(filepaths, number_images)
        for i, filepath in enumerate(filepaths):
            img = Image.open(filepath)
            resizing_factor = 1 / grid_size * resize_scale
            if resizing_factor < 1.0: # Ensures that images are not upsampled
                new_size = tuple(int(dim * resizing_factor) for dim in img.size)
                img = img.resize(new_size)
            imgs.append(img)        
    
    # Create grid
    size = imgs[0].size
    w, h = size
    grid = Image.new("L", size=(w * grid_size, h * grid_size)) # L for grayscale
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        print(f"\radding images to grid: {i + 1}/{len(imgs)}", flush=True, end="")
        col = i % grid_size
        row = i // grid_size
        paste_position = (col * w, row * h)
        grid.paste(img, paste_position)
    grid.save(f"{file_prefix}_img-{original_size[0]}x{original_size[1]}_grid-{grid_size}x{grid_size}.png", dpi=tuple(int(dpi) for dim in size))
    grid.show()



def load_datasets(hparams):
    print("loading datasets...")
    source_data = load_source_data_for_domain_translation(
        batch_size=hparams["batch_size"], 
        image_size=hparams["image_size"], 
        data_dir=hparams["source_dir"], 
        in_channels=hparams["in_channels"], 
        class_cond=False
    )
    target_data = load_source_data_for_domain_translation(
        batch_size=hparams["batch_size"], 
        image_size=hparams["image_size"], 
        data_dir=hparams["target_dir"], 
        in_channels=hparams["in_channels"], 
        class_cond=False
    )
    batch, _ = list(source_data)[0]
    batch = batch.to(hparams["device"])
    return source_data, target_data, batch



def load_model_diffusion(hparams):
    print("setting up diffusion specs...")
    model_params = dict(
        model_path=f"models/{hparams['image_size']}x{hparams['image_size']}_diffusion.pt", 
        attention_resolutions="32,16,8",
        diffusion_steps=1000,
        image_size=hparams["image_size"],
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True, 
        in_channels=hparams["in_channels"]
    )
    defaults = model_and_diffusion_defaults()
    defaults.update(model_params)
    args = Namespace(**defaults)
    print(*defaults.items(), sep="\n")

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(hparams["device"])
    model.convert_to_fp16()
    model.eval()
    
    return model, diffusion



def encode(hparams, batch):
    # Use DDIM to encode to latents
    print("encoding the source images...")
    start = time.time()
    noise = diffusion.ddim_reverse_sample_loop(
        source_model,
        batch,
        clip_denoised=False,
        device=hparams["device"], 
    )
    end = time.time()
    print(f"obtained latent representation for {batch.shape[0]} samples...")
    print(f"encoding step takes {end - start} seconds")
    return noise



def decode(hparams, noise):
    # Next, decode the latents to the target class.
    # We use smaller batches to solve GPU memory issues.
    print("decoding the latents...")
    sample_list = list()
    microbatch = 2

    index = 0
    while microbatch * index < batch_size:
        start = time.time()
        start_index = microbatch * index
        end_index = min(batch_size, microbatch * (index + 1))

        sample = diffusion.ddim_sample_loop(
            model,
            (end_index - start_index, 3, image_size, image_size),
            noise=noise[start_index:end_index],
            clip_denoised=True,
            device=hparams["device"],
            eta=0
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample_list.append(sample)

        index += 1
        end = time.time()
        print(f"decoding step, batch {index} takes {end - start} seconds")

    sample = torch.cat(sample_list, axis=0)
    return sample



if __name__ == "__main__":

    # -------------------------------
    # ARGUMENTS
    # -------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_directory", type=str, help="Path to source dataset (default 'models/US_simulated_mini_100')", dest="source_dir", default="./models/US_simulated_mini_100")
    parser.add_argument("-t", "--target_directory", type=str, help="Path to target dataset (default 'models/US_real_mini_100')", dest="target_dir", default="./models/US_real_mini_100")
    parser.add_argument("-m", "--montage", type=int, help="Size of montage (default 0)", default=0)
    args = parser.parse_args()

    # -------------------------------
    # HPARAMS
    # -------------------------------
    
    hparams = dict(
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        source_dir = args.source_dir, 
        target_dir = args.target_dir, 
        batch_size = 2, 
        image_size = 512, # 64, 256, 512
        in_channels = 3, # Not functional with 1 for some reason
    )
    print("hparams:")
    print(*hparams.items(), sep="\n")

    # -------------------------------
    # DATASET
    # -------------------------------
    
    if args.montage > 0:
        montage(args.montage, dataset_directory=args.source_dir, resize_scale=3.0)
        montage(args.montage, dataset_directory=args.target_dir, resize_scale=3.0)

    source_data, target_data, batch = load_datasets(hparams)
    
    if args.montage > 0:
        montage(args.montage, dataset_directory=args.source_dir, dataloader=source_data, resize_scale=3.0)
        montage(args.montage, dataset_directory=args.target_dir, dataloader=target_data, resize_scale=3.0)
    
    # -------------------------------
    # SOURCE MODEL
    # -------------------------------
    
    source_model, source_diffusion = load_model_diffusion(hparams)
    
    # Training ...
    
    # -------------------------------
    # TARGET MODEL
    # -------------------------------
    
    target_model, target_diffusion = load_model_diffusion(hparams)
    
    # Training ...
    
    # -------------------------------
    # SOURCE-TARGET TRANSLATION
    # -------------------------------
    
    noise = encode(hparams, batch)
    translated = decode(hparams, noise)
    
    # Save the samples as PIL images
    target_images = list()
    for i in range(translated.shape[0]):
        image = Image.fromarray(translated[i].cpu().numpy())
        target_images.append(image)
    print(f"created {len(target_images)} translated samples")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
