import os
import sys
import os

IN_COLAB = False
if 'COLAB_GPU' in os.environ:
    IN_COLAB = True

cwd = os.getcwd()

if (IN_COLAB):
    MOTSYNTH_ROOT = '/content/gdrive/MyDrive/CVCS/storage/MOTSynth'
    MOTCHA_ROOT = '/content/gdrive/MyDrive/CVCS/storage/MOTChallenge'
    OUTPUT_DIR = '/content/gdrive/MyDrive/CVCS/storage/motsynth_output'
else:
    # windows config
    MOTSYNTH_ROOT = cwd + '\storage\MOTSynth'
    MOTCHA_ROOT = cwd + '\storage\MOTChallenge'
    OUTPUT_DIR = cwd + '\storage\motsynth_output'
