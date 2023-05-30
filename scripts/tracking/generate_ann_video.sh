#!/bin/sh

SOURCE_VIDEO_PATH="c:/Users/Matteo/Desktop/cvcs/storage/tracking_videos/source/MOT17-09.mp4"
TARGET_VIDEO_PATH="c:/Users/Matteo/Desktop/cvcs/storage/tracking_videos/target/MOT17-09_tracking.mp4"
MODEL_PATH="c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"

python -m experiments.tracking.inference_byteTrack_frcnn \
    --source-video $SOURCE_VIDEO_PATH \
    --target-video $TARGET_VIDEO_PATH \
    --model $MODEL_PATH 
    
