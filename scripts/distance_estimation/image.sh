#/bin/bash

INPUT_PATH="path-to-input.png"
OUTPUT_PATH="c:/Users/Matteo/Desktop/cvcs/storage/distance_estimation/report/image"
MODEL_PATH="c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"

python -m experiments.distance_estimation.image --input $INPUT_PATH --output $OUTPUT_PATH --model $MODEL_PATH
