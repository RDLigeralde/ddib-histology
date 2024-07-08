# ---------------------------------
# SCRIPT
# ---------------------------------

SEARCH_DIR=./models/$(basename $DATA_DIR)
MODEL_PATH=$(find $SEARCH_DIR -name *model*$STOP.pt)

ml cuda
ml miniconda3
source activate ddib
python3 scripts/ultrasound_sample.py --model_path $MODEL_PATH $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# Other commands
# squeue -u $USER
# tail -f jobID -n 10000
# scancel jobID
