#/bin/bash

INPUT_PATH="c:/Users/Matteo/Desktop/cvcs/experiments/distance_estimation/pedestrians.mp4"
OUTPUT_PATH="c:/Users/Matteo/Desktop/cvcs/storage/distance_estimation/report/video"
MODEL_PATH="c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"

python -m experiments.distance_estimation.video --input $INPUT_PATH --output $OUTPUT_PATH --model $MODEL_PATH
