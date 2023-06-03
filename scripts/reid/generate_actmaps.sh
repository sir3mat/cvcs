#!/bin/sh

python -m experiments.reid.activation_maps --config-file ./configs/reid/METRIC/actmaps/r18_actmaps.yaml
python -m experiments.reid.activation_maps --config-file ./configs/reid/METRIC/actmaps/r18_fc512_actmaps.yaml
python -m experiments.reid.activation_maps --config-file ./configs/reid/METRIC/actmaps/resnet_custom_actmaps.yaml