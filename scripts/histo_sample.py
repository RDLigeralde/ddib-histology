# -----------------------------------------------------------------------------------------
# @dcfrey
# Adapted for creating simulated WSIs
# Based on https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
# -----------------------------------------------------------------------------------------
"""
Uses a trained diffusion model to create synthetic images from sampled Gaussian noise.
Useful for sanity-checking individual models before attempting style transfer
"""
import torch.distributed as dist
from PIL import Image
import numpy as np
import torch as th

import argparse
import os
import torch 

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults_histology, # ultrasound defaults
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def show_npz(data, directory, in_channels=3, ncols=10):
    directory_singles = os.path.join(directory, f"{len(data)}")
    if not os.path.exists(directory_singles):
        os.makedirs(directory_singles, exist_ok=True)
    data_pad = list()
    pad_width = ((1,1), (1,1), (0,0)) if in_channels == 3 else ((1,1), (1,1))
    for i, arr in enumerate(data):
        img = Image.fromarray(arr) # ndim=3 for RGB, ndim=2 for grayscale
        img.save(os.path.join(directory_singles, f"{i:04d}.png"))
        arr_pad = np.pad(np.array(img), pad_width=pad_width, mode='constant', constant_values=0)
        data_pad.append(arr_pad)
    
    data_pad = np.stack(data_pad, axis=0)  # Stack images along a new dimension
    data_pad = data_pad.transpose(0, 2, 1, 3)
    data_pad = np.concatenate(data_pad, axis=1)  # Concatenate images along the height dimension
    #data_pad = data_pad.reshape(data_pad.shape[0] * data_pad.shape[1], data_pad.shape[2] * data_pad.shape[3], in_channels)
    modes = {1: "L", 3: "RGB"}
    imgs = Image.fromarray(data_pad.squeeze(), mode=modes[in_channels])
    imgs.show()
    imgs.save(os.path.join(directory, f"montage_{len(data)}.png"))

def main(args):

    log_dir = f"{args.model_path}_{args.num_samples}-samples_t-{args.timestep_respacing}_useddim-{args.use_ddim}"
    logger.configure(log_dir, use_datetime=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults_histology().keys()) # ultrasound defaults
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device='cuda'
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size), # (B, C, H, W)
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if True:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        show_npz(arr, log_dir, args.in_channels)

    # dist.barrier()
    logger.log("Sampling complete")

def create_argparser():
    defaults = dict()
    defaults.update(model_and_diffusion_defaults_histology()) # ultrasound defaults
    parser = argparse.ArgumentParser()
    # Sampling Defaults
    parser.add_argument("--clip_denoised",type=bool,default=True)
    parser.add_argument("--num_samples",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--use_ddim",type=bool,default=True)
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--experiment",type=str,default='none')
    # Training Defaults
    parser.add_argument("--log_dir",type=str,default="./models/")
    parser.add_argument("--resume_checkpoint",type=str,default="")
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--microbatch",type=int,default=-1)
    parser.add_argument("--log_interval",type=int,default=100)
    parser.add_argument("--save_interval",type=int,default=1000)
    parser.add_argument("--ema_rate",type=float,default=0.9999)
    parser.add_argument("--schedule_sampler",type=str,default="uniform")
    parser.add_argument("--fp16_scale_growth",type=float,default=1e-3)
    parser.add_argument("--weight_decay",type=float,default=0.0)
    parser.add_argument("--lr_anneal_steps",type=int,default=0)
    parser.add_argument("--stop",type=int,default=10000)
    parser.add_argument("--data_dir",type=str,default="")
    parser.add_argument("--model_out",type=str,default="./models/")
    add_dict_to_argparser(parser, defaults)
    return parser


class Experiment:
    def __init__(self, args):
        self.args = args
        if args.experiment == "steps":
            self.compare_steps()
        if args.experiment == "ddim":
            self.compare_ddim()
        if args.experiment == "timesteps":
            self.compare_timesteps()
        if args.experiment == "none":
            main(args)
        
    def compare_steps(self):
        args = self.args
        model_basename = os.path.basename(args.model_path)
        model_dirname = os.path.dirname(args.model_path)
        print(f"MODEL BASENAME: {model_basename}")
        print(f"MODEL DIRNAME: {model_dirname}")
        for steps in range(0, (int(args.stop) + int(args.save_interval)), int(args.save_interval)):
            print(args.stop, steps)
            print(type(args.stop), type(steps))
            print(f'replace(f"{int(args.stop):06d}", f"{steps:06d}")')
            new_model_basename = model_basename.replace(f"{int(args.stop):06d}", f"{steps:06d}")
            print(f"NEW MODEL BASENAME: {new_model_basename}")
            model_path = os.path.join(model_dirname, new_model_basename)
            print(f"MODEL PATH: {model_path}")
            args.model_path = model_path
            main(args)
    
    def compare_ddim(self):
        args = self.args
        for use_ddim in [True, False]:
            args.use_ddim = use_ddim
            main(args)
    
    def compare_timesteps(self):
        args = self.args
        for t in range(100, int(args.diffusion_steps), 100):
            args.timestep_respacing = str(t)
            main(args)
            #args.timestep_respacing = f"ddim{t}"
            #main(args)

if __name__ == "__main__":
    args = create_argparser().parse_args()
    print("arguments: ")

    for key, value in vars(args).items():
        print(f"- {key} : {value}")
    
    if args.experiment == "none":
        main(args)
    if args.experiment != "none":
        Experiment(args)

