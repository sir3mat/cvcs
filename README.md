# Vision-Based pedestrian analysis system

This repository provides the implementation of our paper "NAME OF PAPER" delivered for the "Computer Vision and Cognitive System" course @[UNIMORE](https://www.unimore.it/)

## Installation

N.B.: Installation only avaiable in win64 environments

Create and activate an environment with all required packages:

```
conda create --name pedestrian_detector --file deps/wins/conda_requirements.txt
conda activate pedestrian_detector
pip install -r deps/win/pip_requirements.txt
```

## Dataset download and preparation:

### Prepare MOTSynth dataset

1. Download MOTSynth_1.

```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_1.zip
unzip ./storage/MOTSynth/MOTSynth_1.zip -d ./storage/MOTSynth/
rm ./storage/MOTSynth/MOTSynth_1.zip
```

2. Delete video from 123 to 256
3. Extract frames from the videos

```
python tools/anns/to_frames.py --motsynth-root ./storage/MOTSynth

# now you can delete other videos
rm -r ./storage/MOTSynth/MOTSynth_1
```

4. Download and extract annotations

```
wget -P ./storage/MOTSynth https://motchallenge.net/data/MOTSynth_coco_annotations.zip
unzip ./storage/MOTSynth/MOTSynth_coco_annotations.zip -d ./storage/MOTSynth/
rm ./storage/MOTSynth/MOTSynth_coco_annotations.zip
```

5. Prepare combined annotations for MOTSynth from the original COCO annotations

```
python tools/anns/combine_anns.py --motsynth-path ./storage/MOTSynth
```

6. Prepare motsynth ouput dir for training results

```
mkdir ./storage/motsynth_output
```

### Prepare MOT17 dataset

1. Download MOT17

```
wget -P ./storage/MOTChallenge https://motchallenge.net/data/MOT17.zip
unzip ./storage/MOTChallenge/MOT17.zip -d ./storage/MOTChallenge
rm ./storage/MOTChallenge/MOTSynth_1.zip
```

2. Generate COCO format annotations

```
python tools/anns/motcha_to_coco.py --data-root ./storage/MOTChallenge
```

### Download pretrained models folder from GDrive

You can find all pretrained models [here](https://drive.google.com/drive/folders/15Lv40x3MquSnKbI4U5aGSZtqQuEmiwMH?usp=share_link) (download them and paste the .pth files in storage/pretrained_models directory).

### Storage directory tree

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
    ├── pretrained_models
```

## Modules

1.  Detection\
    See [docs/DETECTOR.md](docs/DETECTOR.md)
2.  Tracking\
    See [docs/TRACKER.md](docs/TRACKER.md)
3.  Retrieval/Re-id from images\
    See [docs/REID.md](docs/REID.md)
4.  Distance violation detector\
    See [docs/DISTANCE_VIOLATION_DETECTOR.md](docs/DISTANCE_VIOLATION_DETECTOR.md)
## Colab Usage

You can also use [Google Colab](https://colab.research.google.com) if you need remote resources like GPUs.
In the notebook folder you can find some useful notebook files and remember to load all the storage folder in your GDrive before usage (N.B. you need at least 150/200 GB).
