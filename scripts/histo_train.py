# -----------------------------------------------------------------------------------------
# @dcfrey
# Adapted for training on histological imaging domains
# Based on https://github.com/openai/guided-diffusion/blob/main/scripts/image_train.py
# -----------------------------------------------------------------------------------------
"""
Train a diffusion model on images.
"""

import argparse
import os
import torch 

from guided_diffusion import dist_util, logger
from dataloading.patch_dataset import load_qpb
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_histology,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def write_arg_dict(log_dir, arg_dict, training_params):
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        f.write("Model Params:")
        for k, v in arg_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTraining Params:")
        for k, v in training_params.items():
            f.write(f"{k}: {v}\n")

def train(args: argparse.Namespace):
    """
    Single-domain diffusion training script
    See parser help text for argument descriptions

    Args:
        args (argparse.Namespace): arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_out):
        os.makedirs(args.model_out)

    log_dir = args.model_out
    logger.configure(log_dir, use_datetime=False)
    print(f'Writing out to {log_dir}')

    logger.log("Creating model and diffusion...")
    arg_dict = args_to_dict(args, model_and_diffusion_defaults_histology().keys())
    model, diffusion = create_model_and_diffusion(
        **arg_dict
    )
    train_keys  = [
        'lr', 
        'batch_size', 
        'microbatch', 
        'log_interval', 
        'save_interval', 
        'ema_rate', 
        'fp16_scale_growth', 
        'weight_decay', 
        'lr_anneal_steps', 
        'stop'
    ]

    training_params = {k: v for k, v in vars(args).items() if k in train_keys}
    write_arg_dict(log_dir, arg_dict, training_params)
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loader...")
    data = load_qpb(
        wsi_dir=args.wsi_dir,
        coord_dir=args.h5_dir,
        batch_size=args.batch_size
    )

    logger.log("Training...")
    try:
        TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        stop=args.stop,
        device=device
        ).run_loop()
    except torch.cuda.OutOfMemoryError:
        logger.log("Memory limit exceeded")
        allocated = torch.cuda.memory_allocated() / (1024**2)
        cached = torch.cuda.memory_reserved() / (1024**2)
        logger.log(f"Allocated: {allocated:.2f} MB")
        logger.log(f"Cached: {cached:.2f} MB")
        torch.cuda.empty_cache()

    
def create_argparser():
    defaults = dict()
    defaults.update(model_and_diffusion_defaults_histology()) # model and diffusion defaults

    parser = argparse.ArgumentParser()
    
    # I/O
    parser.add_argument("--model_out",type=str,default="./models/", help="Model output dir")
    parser.add_argument("--log_dir",type=str,default="./models/", help="Log output dir")
    parser.add_argument("--resume_checkpoint",type=str,default="", help="Training resume point")
    parser.add_argument("--log_interval",type=int,default=100, help="Log interval")
    parser.add_argument("--save_interval",type=int,default=1000, help="Save interval")
    parser.add_argument("--wsi_dir",type=str, help="Path to WSI directory")
    parser.add_argument("--h5_dir",type=str, help="Path to H5 directory")

    # Training Basics
    parser.add_argument("--lr",type=float,default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size",type=int,default=1, help="Batch size")
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
    train(args)

if __name__ == "__main__":
    main()
