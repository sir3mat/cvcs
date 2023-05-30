#/bin/bash

INPUT_PATH=""
OUTPUT_PATH=""
MODEL_PATH=""

python -m experiments.distance_estimation.video --input $INPUT_PATH --output $OUTPUT_PATH --model $MODEL_PATH
