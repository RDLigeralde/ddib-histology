#!/bin/bash
#BSUB -J ddib_translate_vid
#BSUB -o ddib_translate_vid.%J.out
#BSUB -e ddib_translate_vid.%J.err
#BSUB -q kimgpu
#BSUB -gpu "num=4"
#BSUB -N matthew.lee1@pennmedicine.upenn.edu

# MODEL
MODEL_FLAGS="--image_size 512 --num_channels 256 --num_res_blocks 3 --in_channels 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

# TRAIN
STOP=10000
TRAIN_FLAGS="--log_dir './models/' --resume_checkpoint '' --lr 1e-5 --batch_size 4 --microbatch -1 --log_interval 100 --save_interval 500 --ema_rate 0.9999 --fp16_scale_growth 1e-3 --weight_decay 0.0 --lr_anneal_steps 0 --stop $STOP"

# SAMPLE
SAMPLE_FLAGS="--num_samples 10 --use_ddim True --experiment 'none'"

# I/O
DATA_DIR="./datasets/histology/bright/"
MODEL_PATH_SOURCE='./models/bright/model009500.pt'
MODEL_PATH_TARGET='./models/dim/model009500.pt'
IMAGE_OUT="./models/bright/translate/video/"

CUDA_VISIBLE_DEVICES=3 python3 -u scripts/ultrasound_translation.py \
--data_path_source $DATA_DIR \
--model_path_source $MODEL_PATH_SOURCE \
--model_path_target $MODEL_PATH_TARGET \
--image_out $IMAGE_OUT \
$MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $TRAIN_FLAGS

