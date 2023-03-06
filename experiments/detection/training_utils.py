import os.path as osp
import os
from torchinfo import summary
import torch
import torch.utils.data
from core.detection.graph_utils import save_train_loss_plot
import core.detection.vision.utils as utils
import coloredlogs
import logging
coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)


def create_optimizer(model, lr, momentum, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(optimizer, lr_scheduler_type, lr_steps, lr_gamma, epochs):
    if lr_scheduler_type == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_steps, gamma=lr_gamma)
        logger.debug(
            f"lr_scheduler: {lr_scheduler_type}, milestones: {lr_steps}, gamma: {lr_gamma}")

    elif lr_scheduler_type == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        logger.debug(
            f"lr_scheduler: {lr_scheduler_type}, T_max: {epochs}")
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler_type}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    return lr_scheduler


def save_model_checkpoint(model, optimizer, lr_scheduler, epoch, scaler, output_dir, args):
    if output_dir:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(
            output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(
            output_dir, "checkpoint.pth"))


def save_plots(losses_dict, batch_loss_dict, output_dir):
    if not losses_dict:
        for name, metric in batch_loss_dict.items():
            losses_dict[name] = []

    for name, metric in batch_loss_dict.items():
        losses_dict[name].extend(metric)
    save_train_loss_plot(losses_dict, output_dir)


def save_model_summary(model, output_dir, batch_size):
    with open(osp.join(output_dir, "summary.txt"), 'w', encoding="utf-8") as f:
        print(summary(model,
                      # (batch_size, color_channels, height, width)
                      input_size=(batch_size, 3, 1080, 1920),
                      verbose=0,
                      col_names=["input_size", "output_size",
                                 "num_params", "kernel_size", "trainable"],
                      col_width=20,
                      row_settings=["var_names"]), file=f)


def save_args(output_dir, args):
    with open(osp.join(output_dir, "args.txt"), 'w', encoding="utf-8") as f:
        print(args, file=f)


def save_evaluate_summary(stats, output_dir):
    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    # the standard metrics
    results = {
        metric: float(stats[idx] *
                      100 if stats[idx] >= 0 else "nan")
        for idx, metric in enumerate(metrics)
    }
    with open(osp.join(output_dir, "evaluate.txt"), 'w', encoding="utf-8") as f:
        print(results, file=f)
