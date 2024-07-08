#!/bin/bash
#BSUB -J ddib_train_bright
#BSUB -o ddib_train_bright.%J.out
#BSUB -e ddib_train_bright.%J.err
#BSUB -q kimgpu
#BSUB -gpu "num=4"
#BSUB -N roblig22@sas.upenn.edu

# MODEL
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --in_channels 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --learn_sigma True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

# TRAIN
STOP=10000
TRAIN_FLAGS="--log_dir './models/' --resume_checkpoint '' --lr 1e-5 --batch_size 1 --microbatch -1 --log_interval 100 --save_interval 5000 --ema_rate 0.9999 --fp16_scale_growth 1e-3 --weight_decay 0.0 --lr_anneal_steps 0 --stop $STOP"

# I/O
H5_DIR='/home/roblig22/ddib/datasets/histology/tilings/bright/patches'
WSI_DIR='/project/kimlab_hshisto/WSI/bright'
MODEL_DIR='/home/roblig22/ddib/models/histo_bright'

# ---------------------------------
# SCRIPT
# ---------------------------------
CUDA_VISIBLE_DEVICES=2 python3 -u scripts/histo_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --wsi_dir $WSI_DIR --h5_dir $H5_DIR --model_out $MODEL_DIR