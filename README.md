# School in AI Project Work

This repository contains the code to train and evaluate a pedestrian detector for 
the "School in Ai 2° edition"@[@UNIMORE](https://www.unimore.it/)

## Installation

N.B.: Installation only avaiable in win64 environments

Create and activate an environment with all required packages:

```
conda create --name ped_detector --file deps/wins/conda_environment.txt
# or conda env create -f deps/win/conda_environment.yml
conda activate cvcspw
pip install -r deps/win/pip_requirements.txt
```

## Dataset download and preparation:
### Solution 1 - From Google Drice
Download the storage folder directly from Google Drive [here](link google drive)
and place it in the root dir of the project
After runnning this step, your storage directory should look like this:
```text
storage
    ├── MOTChallenge
        ├── MOT17
        ├── motcha_coco_annotations
    ├── MOTSynth
        ├── annotations
        ├── comb_annotations
        ├── frames
    ├── motsynth_output
```
### Solution 2 - From scratch
#### Prepare MOTSynth dataset
1. Download MOTSynth_1.
```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_1.zip
unzip ./storage/MOTSynth/MOTSynth_1.zip
rm ./storage/MOTSynth/MOTSynth_1.zip
```
2. Delete video from 123 to 256
2. Extract frames from the videos
```
python tools/anns/to_frames.py --motsynth-root ./storage/MOTSynth

# now you can delete other videos
rm -r ./storage/MOTSynth/MOTSynth_1
```
3. Download and extract annotations
```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_coco_annotations.zip
unzip ./storage/MOTSynth/MOTSynth_coco_annotations.zip
rm ./storage/MOTSynth/MOTSynth_coco_annotations.zip
```
4. Prepare combined annotations for MOTSynth from the original coco annotations
```
python tools/anns/combine_anns.py --motsynth-path ./storage/MOTSynth
```
#### Prepare MOT17 dataset


## Colab Usage

You can also use [Google Colab](https://colab.research.google.com) if you need remote resources like GPUs.
In the notebook folder you can find some useful .ipynb files and remember to load the storage folder in your GDrive before usage.

## Object Detection

An adaption of torchvision's detection reference code is done to train Faster R-CNN on a portion of the MOTSynth dataset. To train the model you can run:
```
./scripts/train_detector
```

