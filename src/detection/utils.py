from torchvision import transforms
from typing import DefaultDict
import matplotlib.pyplot as plt
import matplotlib
import torch
import logging
from torchvision.utils import draw_bounding_boxes
matplotlib.style.use('ggplot')


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


def show_bbox(img, output, th=None):
    img_to_show = torch.clip(img*255, 0, 255)
    img_to_show = img_to_show.type(torch.uint8)
    if th == None:
        img_with_bbbox = draw_bounding_boxes(
            img_to_show, boxes=output, width=4)
    else:
        img_with_bbbox = draw_bounding_boxes(
            img_to_show, boxes=output["boxes"][output['scores'] > th], width=4)
    plot_img_tensor(img_with_bbbox)
