#!/bin/sh

python -m experiments.reid.main_reid --config-file ./configs/reid/train/METRIC/r18_fc512_motsynth_train.yaml
python -m experiments.reid.main_reid --config-file ./configs/reid/train/METRIC/r18_motsynth_train.yaml
python -m experiments.reid.main_reid --config-file ./configs/reid/train/METRIC/resnet_custom_deeper_motsynth_train.yaml