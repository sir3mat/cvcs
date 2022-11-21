#!/bin/sh

python -m  tools.experiments.inference_detector  --model-path "path-to-model.pth" --threshold 0.8 --input ./storage/MOTChallenge/MOT17/test/MOT17-03-FRCNN/img1/000004.jpg