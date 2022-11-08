# School in AI Project Work

Questa repository contiene il codice utilizzato per effettuare training e analisi di un pedestrian detector per il corso "School in Ai 2° edizione"@[@UNIMORE](https://www.unimore.it/)

## Installazione

N.B.: Installation only avaiable in win64 environments

Create and activate an environment with all required packages:

```
conda create --name cvcspw --file deps/wins/conda_environment.txt
# or conda env create -f deps/win/conda_environment.yml

conda activate cvcspw

# in case of error with cuda
# conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r deps/win/pip_requirements.txt
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

### Dataset annotation format

MOT annotations in `/storage/mot_annotations/frame/gt/gt.txt` follow this format:

```text
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <vis>, <x>, <y>, <z>
```

## Colab Usage

You can also use [Google Colab](https://colab.research.google.com) if you need remote resources like GPUs.
In the notebook folder you can find some useful .ipynb files and remember to load the storage folder in your GDrive before usage.

## Object Detection

We adapt torchvision's detection reference code to train Faster R-CNN on a portion of the MOTSynth dataset. To train Faster R-CNN with a ResNet50 with FPN backbone, you can run the following:

```
python -m  tools.train_detector --epochs 10
```

