import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import torch as th
import numpy as np
import cv2

from pathlib import Path
import argparse
import shutil
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_histology,
    classifier_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataloading.patch_dataset import qpb_one_shot

def translate(args: argparse.Namespace):
    """
    Performs DDIB domain translation using two trained models.
    See parser help text for argument descriptions.

    Args:
        args (argparse.Namespace): arguments
    """
    logger.log(f"arguments: {args}")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    if not os.path.exists(args.image_out):
        os.makedirs(args.image_out)
    logger.configure(args.image_out, use_datetime=False)

    logger.log("Creating model and diffusion...")
    arg_dict  = args_to_dict(args, model_and_diffusion_defaults_histology().keys())
    model, diffusion = create_model_and_diffusion(
        **arg_dict
    )

    logger.log('Sending model to GPU...')
    model = model.to(device)

    logger.log('Loading in data...')
    data = qpb_one_shot(
        wsi_dir=args.wsi_dir,
        coord_dir=args.h5_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    logger.log("Running image translation...")
    for i, (batch, extra) in enumerate(data):
        logger.log(f"Translating batch {i}, shape {batch.shape}.")

        batch = batch.to(device)
        logger.log('Loading source model...')
        model.load_state_dict(
            th.load(args.model_path_source, map_location=device)
        )
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("Encoding source along with intermediates...")
        noise_interm = diffusion.ddim_reverse_sample_loop_progressive(
            model,
            batch,
            clip_denoised=False,
            model_kwargs=None,
            device=device
        )
        noise_interm_list = []
        for sample in noise_interm:
            noise_interm_list.append(sample["sample"])
        noise = noise_interm_list[-1]

        logger.log('Loading target model...')
        model.load_state_dict(
            th.load(args.model_path_target, map_location=device)
        )
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        sample_interm = diffusion.ddim_sample_loop_progressive(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            device=device,
            eta=args.eta
        )
        sample_interm_list = []
        for sample_i in sample_interm:
            sample_interm_list.append(sample_i["sample"])
        
        sample = sample_interm_list[-1]

        logger.log('Processing the output for saving...')
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        images = [sample.cpu().numpy()]
        logger.log(f"created {len(images) * args.batch_size} samples")
        logger.log(f"image is of shape {images[0].shape}")

        logger.log("Saving translated images.")
        images = np.concatenate(images, axis=0)

        for index in range(images.shape[0]):
            filepath = os.path.join(args.image_out, f"final_image_{index}.png")
            image = Image.fromarray(images[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

            filename = os.path.join(args.image_out, f'transition_video_{index}.mp4')
            fps = 450
            frame_size = (args.image_size, args.image_size)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)

            for step in range(len(noise_interm_list)):
                noise_image = preprocess_image(noise_interm_list[step][index])
                writer.write(noise_image)

            for step in range(1, len(sample_interm_list)):
                sample_image = preprocess_image(sample_interm_list[step][index])
                writer.write(sample_image)
                
            writer.release()

    logger.log(f"Translation complete (っ◕‿◕)っ")

def preprocess_image(image_tensor):
    """
    Apply necessary preprocessing steps to each image tensor.
    """
    image = ((image_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
    image = image.permute(1, 2, 0)  # C x H x W -> H x W x C
    return image.contiguous().cpu().numpy()

def create_argparser():
    defaults = dict()
    defaults.update(model_and_diffusion_defaults_histology())
    defaults.update(classifier_defaults())
    
    parser = argparse.ArgumentParser()
    # TRANSLATION DEFAULTS
    parser.add_argument("--model_path_source",type=str,default="",help="Path to source model")
    parser.add_argument("--model_path_target",type=str,default="",help="Path to target model")
    parser.add_argument("--wsi_dir",type=str,default="",help="Path to source WSIs")
    parser.add_argument("--h5_dir",type=str,default="",help="Path to source h5s")
    parser.add_argument("--image_out",type=str,default="",help="Path to save translated images")
    parser.add_argument("--eta",type=float,default=0.0,help="Diffusion eta")
    parser.add_argument("--resume_checkpoint",type=str,default="", help="Training resume point")
    parser.add_argument("--log_interval",type=int,default=100, help="Log interval")
    parser.add_argument("--save_interval",type=int,default=1000, help="Save interval")

    # Sampling Defaults
    parser.add_argument("--clip_denoised",type=bool,default=True)
    parser.add_argument("--num_samples",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--use_ddim",type=bool,default=True)
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--experiment",type=str,default=None)

    # Training Defaults
    parser.add_argument("--log_dir",type=str,default="./models/")
    parser.add_argument("--lr",type=float,default=1e-5, help="Learning rate")
    parser.add_argument("--microbatch",type=int,default=-1, help="Microbatch size")
    parser.add_argument("--ema_rate",type=float,default=0.9999, help="EMA rate")
    parser.add_argument("--schedule_sampler",type=str,default="uniform", help="Schedule sampler")
    parser.add_argument("--fp16_scale_growth",type=float,default=1e-3, help="FP16 scale growth")
    parser.add_argument("--weight_decay",type=float,default=0.0, help="Weight decay")
    parser.add_argument("--lr_anneal_steps",type=int,default=0, help="Learning rate anneal (decay) steps")
    parser.add_argument("--stop",type=int,default=10000, help="Max iteration count")

    # Model Params
    parser.add_argument("--image_size", type=int, help="Image size")
    parser.add_argument("--in_channels", type=int, help="Number of input channels")
    parser.add_argument("--num_channels", type=int, help="Number of input channels")
    parser.add_argument("--channel_mult", type=str, help="Channel multipliers") # WHY WOULD YOU MAKE THIS A STRING????
    parser.add_argument('--num_res_blocks', type=int, help="Number of residual blocks")
    parser.add_argument('--attention_resolutions', type=str, help="Attention resolutions") # WHY WOULD YOU MAKE THIS A STRING????
    parser.add_argument('--num_heads', type=int, help="Number of attention heads")

    # Diffusion Params
    parser.add_argument("--noise_schedule",type=str, help="Noise schedule")
    parser.add_argument("--diffusion_steps",type=int, help="Diffusion steps")
    
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    parser = create_argparser()
    args = parser.parse_args()
    translate(args)

if __name__ == "__main__":
    main()
