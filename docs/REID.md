# Person re-identification

Pedestrian reid module is performed using the [torchreid](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid) implementation to train our model. 

# How to use
An adaption of torchvision's detection reference code is done to train Faster R-CNN on a portion of the MOTSynth dataset.

- To generate reid images and the csv files necessary for training:

```
mkdir ./csv_anns

python ./experiments/reid/data_prep/crop_mots.py --ann-path ./storage/MOTSynth/comb_annotations/train_mini.json --csv-path ./csv_anns/motsynth.csv

python ./tools/anns/store_reid_imgs.py --ann-path ./storage/MOTChallenge/motcha_coco_annotations/MOT17-train.json
python ./experiments/reid/data_prep/data.py --anns-dir ./storage/MOTChallenge/motcha_coco_annotations --imgs-dir ./storage/MOTChallenge/MOT17 --csv-dir ./csv_anns
```

- To prepare the environment for torchreid:

```
cd deep-person-reid/

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

python setup.py develop

```

- To run training and test on MOTSynth and MOT17 datasets, respectively:

```
cd ..

python ./experiments/reid/reid.py --csv-mots ./csv_anns/motsynth.csv --csv-mot17 ./csv_anns/MOT17_tr.csv --img-mots ./storage/MOTSynth/reid --img-mot17 ./storage/MOTChallenge/reid
```

