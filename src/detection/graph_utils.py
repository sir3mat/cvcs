import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from typing import DefaultDict
import matplotlib.pyplot as plt
import matplotlib
import torch
import logging
from torchvision.utils import draw_bounding_boxes
matplotlib.style.use('ggplot')
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)


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


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(nrows=len(imgs), ncols=1,
                            figsize=(45, 21), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img = np.asarray(img)
        axs[i, 0].imshow(img)
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def plot_img_tensor(img_tensor):
    transforms.ToPILImage()(img_tensor).show()


def show_img(data_loader, model, device, th=0.7):
    for imgs, target in data_loader:
        with torch.no_grad():
            prediction = model([imgs[0].to(device)])[0]
        plot_img_tensor(add_bbox(imgs[0], prediction, th))
        plot_img_tensor(add_bbox(imgs[0], target[0]['boxes']))
        break


def add_bbox(img, output, th=None):
    img_canvas = img.clone()
    img_canvas = torch.clip(img*255, 0, 255)
    img_canvas = img_canvas.type(torch.uint8)

    if th == None:
        img_with_bbbox = draw_bounding_boxes(
            img_canvas, boxes=output, width=4)
    else:
        scores_list = [score for score in (
            output["scores"][output["scores"] > th]).tolist()]
        labels_list = [str(label) for label in (
            output["labels"][output["scores"] > th]).tolist()]
        labels = ["person" for label in labels_list if label == "1"]
        for i in range(0, len(scores_list)):
            labels[i] = f"{labels[i]}:{scores_list[i]:.3f}"
        img_with_bbbox = draw_bounding_boxes(
            img_canvas, boxes=output["boxes"][output['scores'] > th], labels=labels, width=4)
    return img_with_bbbox
