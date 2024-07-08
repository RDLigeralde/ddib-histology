# ---------------------------------
# SCRIPT
# ---------------------------------

ml cuda
ml miniconda3
source activate ddib
#mpiexec -n $NUM_GPUS python3 scripts/ultrasound_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
python3 scripts/ultrasound_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# Other commands
# squeue -u $USER
# tail -f jobID -n 10000
# scancel jobID
