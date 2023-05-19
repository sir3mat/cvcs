# Person re-identification

Pedestrian reid module is performed using the [torchreid](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid) implementation to train our model.

# How to use

1. To prepare the environment for torchreid:

```
cd deep-person-reid/

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

python setup.py develop

```

2 To run training and test on MOTSynth and MOT17 datasets, respectively:

```
cd ..

./scripts/reid/training.sh
```

## TensorBoard

The SummaryWriter() for tensorboard will be automatically initialized in engine.run() when you are training your model. Therefore, you do not need to do extra jobs. After the training is done, the _tf.events_ file will be saved in save_dir. Then, you just call in your terminal:

```
pip install tensorflow tensorboard
tensorboard --logdir=your_save_dir
```

Access tensorboard visiting http://localhost:6006/ in a web browser.
See pytorch tensorboard for further information.
