# Person re-identification

Pedestrian reid module is performed using the [torchreid](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid) implementation to train our model. 

# How to use

- To generate reid images and the csv files necessary for training:

```
mkdir ./csv_anns

python ./experiments/reid/data_prep/crop_mots.py --ann-path ./storage/MOTSynth/comb_annotations/train_mini.json --csv-path ./csv_anns/motsynth.csv

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

python ./experiments/reid/reid.py --csv-mots ./csv_anns/motsynth.csv  --img-mots ./storage/MOTSynth/reid 
```

