# Exploring the Potential of Synthetic Data for Pedestrian Analysis

This repository provides the implementation of our paper "Exploring the Potential of Synthetic Data for Pedestrian Analysis" delivered for the "Computer Vision and Cognitive System" course @[UNIMORE](https://www.unimore.it/)

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
python tools/anns/combine_anns.py --motsynth-path ./storage/MOTSynth --split motsynth_split3

```

6. Prepare reid images

```
python tools/anns/store_reid_imgs.py --ann-path ./storage/MOTSynth/comb_annotations/motsynth_split3.json --frames-path ./storage/MOTSynth
```

7. Prepare motsynth ouput dir for training results

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
python ./tools/anns/motcha_to_coco.py --data-root storage/MOTChallenge --dataset MOT17 --split train
```

3. Generate reid images

```
python tools/anns/store_reid_imgs.py --ann-path ./storage/MOTChallenge/motcha_coco_annotations/MOT17-train.json
```

### Download pretrained models folder from GDrive

You can find all pretrained models for detection and reid [here](https://drive.google.com/drive/folders/1RiVywWYQA6XhhPIntThI1LrT-Up-_eI9?usp=sharing) (download them and paste the .pth files in storage/pretrained_models directory).

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
3.  Retrieval/Re-id\
    See [docs/REID.md](docs/REID.md)
4.  Distance violation detector\
    See [docs/DISTANCE_VIOLATION_DETECTOR.md](docs/DISTANCE_VIOLATION_DETECTOR.md)

## Assets

Some images, videos and plots are available [here](https://drive.google.com/drive/folders/1IAz45YagXviQ8C3J4zDzPjf0cBto5KzU?usp=share_link)

# Authors
- [Sirri Matteo](https://github.com/sir3mat)
    - email: 254179@studenti.unimore.it
- [Manghi Ilaria](https://github.com/ilariamanghi)
    - email: 244770@studenti.unimore.it
- [Riccardo Benini](https://github.com/RiccardoBenini98) 
    - email: 244321@studenti.unimore.it
