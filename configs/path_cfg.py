import os
import sys
IN_COLAB = 'google.colab' in sys.modules


cwd = os.getcwd()

if(IN_COLAB):
    # colab config
    MOTSYNTH_ROOT = '/content/gdrive/MyDrive/CVCS/storag/MOTSynth'
    MOTCHA_ROOT = '/content/gdrive/MyDrive/CVCS/storage/MOTChallenge'
    OUTPUT_DIR = '/content/gdrive/MyDrive/CVCS/storage/motsynth_output'
else:
    # windows config
    MOTSYNTH_ROOT = cwd + '\storage\MOTSynth'
    MOTCHA_ROOT = cwd + '\storage\MOTChallenge'
    OUTPUT_DIR = cwd + '\storage\motsynth_output'
