#!/bin/sh

python -m experiments.reid.main_reid --config-file ./configs/reid/METRIC/visrank/r18_custom_visrank.yaml
python -m experiments.reid.main_reid --config-file ./configs/reid/METRIC/visrank/r18_fc512_visrank.yaml
python -m experiments.reid.main_reid --config-file ./configs/reid/METRIC/visrank/r18_visrank.yaml