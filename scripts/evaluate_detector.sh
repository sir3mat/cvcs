#!/bin/sh

python -m  tools.train_detector --model-eval "d://cvcspw/storage/motsynth_output/detection_logs/fasterrcnn_training/checkpoint.pth" --test-only
