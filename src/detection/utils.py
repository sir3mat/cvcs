from torchvision import transforms
from collections import defaultdict
from typing import DefaultDict
import matplotlib.pyplot as plt
import matplotlib
import torch
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
matplotlib.style.use('ggplot')   # type: ignore


def save_plot(train_loss_list, label, output_dir):
    """
    Function to save the loss plot to disk.
    """
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss_list, linestyle='-',
        label=label
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/{label}.png")


def save_train_loss_plot(train_loss_dict: DefaultDict, output_dir):
    """
    Function to save the loss plots to disk.
    """
    for key in train_loss_dict.keys():
        save_plot(train_loss_dict[key], key, output_dir)


def plot_img_tensor(img_tensor):
    transforms.ToPILImage()(img_tensor).show()

def show_bbox():
    ...

