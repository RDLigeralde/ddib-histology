#!/bin/bash
#BSUB -J ddib_sample_dim_
#BSUB -o ddib_sample_dim_.%J.out
#BSUB -e ddib_sample_dim_.%J.err
#BSUB -q kimgpu
#BSUB -gpu "num=4"
#BSUB -N matthew.lee1@pennmedicine.upenn.edu

# MODEL
MODEL_FLAGS="--image_size 512 --num_channels 256 --num_res_blocks 3 --in_channels 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --learn_sigma True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

# TRAIN
STOP=10000
TRAIN_FLAGS="--log_dir './models/' --resume_checkpoint '' --lr 1e-5 --batch_size 8 --microbatch -1 --log_interval 100 --save_interval 500 --ema_rate 0.9999 --fp16_scale_growth 1e-3 --weight_decay 0.0 --lr_anneal_steps 0 --stop $STOP"

# I/O
MODEL_PATH='./models/bright/model010000.pt'

# SAMPLE
SAMPLE_FLAGS="--num_samples 10 --use_ddim True --experiment timesteps"

# ---------------------------------
# SCRIPT
# ---------------------------------

CUDA_VISIBLE_DEVICES=2 python3 -u scripts/ultrasound_sample.py --model_path $MODEL_PATH $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
