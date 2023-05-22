#!/bin/sh

RESULTS_FOLDER='c:/Users/Matteo/Desktop/cvcs/storage/mota'
MODEL_PATH="c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"

python -m experiments.tracking.generate_preds_byteTrack_frcnn \
    --results-folder $RESULTS_FOLDER \
    --model $MODEL_PATH
