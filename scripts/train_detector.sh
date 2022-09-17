#!/bin/sh

python -m  tools.train_detector --model fasterrcnn_resnet50_fpn --batch-size 8 --world-size 1 --trainable-backbone-layers 1 --backbone resnet50 --train-dataset train --epochs 10
