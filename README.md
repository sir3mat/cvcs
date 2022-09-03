# CVCS Project Work Group #13

This repository provide the implementation of our paper "NAME OF PAPER" delivered for the "Computer Vision and Cognitive System" course [@UNIMORE](https://www.unimore.it/)

## Installation
Create and activate an environment with all required packages:
```
conda create --name cvcs --file environment.txt
```

## Dataset Download and Preparation:
Download the storage folder content from [here](https://drive.google.com/drive/folders/1rQY3S5DZ2Au5VEPeIFYB2pgph97C_8tl?usp=sharing)

After runnning this step, your storage directory should look like this:
```text
storage
    ├── MOTChallenge
        ├── data
        ├── MOT17
        ├── motcha_coco_annotations
        ├── motcha_reid_images
    ├── MOTSynth
        ├── annotations
        ├── comb_annotations
        ├── frames
        ├── mot_annotations
        ├── mots_annotations
        ├── reid
    ├── motsynth_output
        ├── models

```


## Object Detection
We adapt torchvision's detection reference code to train Faster R-CNN on a portion of the MOTSynth dataset. To train Faster R-CNN with a ResNet50 with FPN backbone, you can run the following:
```
python -m  tools.train_detector --model fasterrcnn_resnet50_fpn\
    --batch-size 5 --world-size 1 --trainable-backbone-layers 1\ 
    --backbone resnet50 --train-dataset train --epochs 10
```

## Multi-Object Tracking
## Distance estimation
## Person Re-Identification

## Acknowledgements
This pipeline is built on top of [motsynth-baselines](https://github.com/dvl-tum/motsynth-baselines) repository.