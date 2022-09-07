import os

cwd = os.getcwd()

if(cwd == '/content'):
    # colab config
    MOTSYNTH_ROOT = cwd + '/gdrive/MyDrive/CVCS/storag/MOTSynth'
    MOTCHA_ROOT = cwd + '/gdrive/MyDrive/CVCS/storage/MOTChallenge'
    OUTPUT_DIR = cwd + '/gdrive/MyDrive/CVCS/storage/motsynth_output'
else:
    # windows config
    MOTSYNTH_ROOT = cwd + '\storage\MOTSynth'
    MOTCHA_ROOT = cwd + '\storage\MOTChallenge'
    OUTPUT_DIR = cwd + '\storage\motsynth_output'
