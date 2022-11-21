#!/bin/sh

python -m  tools.experiments.inference_detector  --model-path ./storage/motsynth_output/detection_logs/frcnn_v1_training_imagenet_motsynth_split3_10epoche/checkpoint.pth --threshold 0.8 --input ./storage/MOTChallenge/MOT17/test/MOT17-03-FRCNN/img1/000004.jpg