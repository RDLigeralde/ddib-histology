#!/bin/bash
#BSUB -J ddib_patch_dim
#BSUB -o ddib_patch_dim.%J.out
#BSUB -e ddib_patch_dim.%J.err
#BSUB -q i2c2_normal
#BSUB -N email

# --------------------
# SCRIPT
# --------------------
python -u datasets/images_from_h5s.py --wsi_dir /project/kimlab_hshisto/WSI/dim --h5_dir "/home/roblig22/ddib/datasets/histology/tilings/dim/patches"