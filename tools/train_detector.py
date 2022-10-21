from configs.path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT, OUTPUT_DIR
import datetime
import os.path as osp
import os
import time
import coloredlogs
import logging
from torchinfo import summary
import torch
import torch.utils.data
from src.detection.vision.mot_data import MOTObjDetect
from src.detection.model_factory import ModelFactory
from src.detection.utils import save_train_loss_plot, show_bbox
import src.detection.vision.presets as presets
import src.detection.vision.utils as utils
from src.detection.vision.engine import train_one_epoch, evaluate
from src.detection.vision.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from src.detection.mot_dataset import get_mot_dataset

coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training", add_help=add_help)

    # Output directory used to save model, plots and summary
    parser.add_argument("--output-dir", default='fasterrcnn_training',
                        type=str, help="Path to save outputs (default: fasterrcnn_training)")

    # Dataset params
    parser.add_argument("--train-dataset", default="motsynth_train",
                        type=str, help="Dataset name. Please select one of the following: motsynth_train, MOT17 (default: motsynth_train)")
    parser.add_argument("--val-dataset", default="motsynth_val",
                        type=str, help="Dataset name. Please select one of the following: motsynth_val, MOT17 (default: motsynth_val)")

    # Transforms params
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="Data augmentation policy (default: hflip)"
    )

    # Data Loaders params
    parser.add_argument(
        "-b", "--batch-size", default=3, type=int, help="Images per gpu (default: 3)"
    )
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="Number of data loading workers (default: 0)"
    )
    parser.add_argument("--aspect-ratio-group-factor", default=3,
                        type=int, help="Aspect ration group factor (default:3)")

    # Model param
    parser.add_argument(
        "--model", default="fasterrcnn_resnet50_fpn", type=str, help="Model name (default: fasterrcnn_resnet50_fpn)")
    parser.add_argument(
        "--weights", default="DEFAULT", type=str, help="Model weights (default: DEFAULT)"
    )
    parser.add_argument(
        "--backbone", default='resnet50', type=str, help="Type of backbone (default: resnet50)"
    )
    parser.add_argument(
        "--trainable-backbone-layers", default=3, type=int, help="Number of trainable layers of backbone (default: 3)"
    )
    parser.add_argument(
        "--backbone-weights", default="DEFAULT", type=str, help="Backbone weights (default: DEFAULT)"
    )

    # Device param
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (default: cuda)")

    # Test mode param
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Optimizer params
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="Learning rate (default: 0.0005)",
    )
    parser.add_argument("--momentum", default=0.9,
                        type=float, metavar="M", help="Momentum (default: 0.9")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="Weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # Lr Scheduler params
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="Name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="Decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="Decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )

    # Mixed precision training params
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")

    # Resume training params
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")

    # training param
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--epochs", default=26, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument("--print-freq", default=20,
                        type=int, help="print frequency")

    return parser


def get_transform(train, data_augmentation):
    if train:
        return presets.DetectionPresetTrain(data_augmentation)
    else:
        return presets.DetectionPresetEval()


def create_datasets(ds_train_name, ds_val_name, data_augmentation):
    dataset_train = dataset_train_no_random = dataset_val = None

    if ds_train_name == "motsynth_train":
        data_path = osp.join(
            MOTSYNTH_ROOT, 'comb_annotations', f"{ds_train_name}.json")
        dataset_train = get_mot_dataset(MOTSYNTH_ROOT,
                                        data_path, transforms=get_transform(True, data_augmentation))
        dataset_train_no_random = get_mot_dataset(MOTSYNTH_ROOT,
                                                  data_path, transforms=get_transform(False, data_augmentation))

    elif ds_train_name == "MOT17":
        train_split_seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
                            'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
        data_path = osp.join(MOTCHA_ROOT, "MOT17", "train")
        dataset_train = MOTObjDetect(data_path, get_transform(
            True, data_augmentation), split_seqs=train_split_seqs)
        dataset_train_no_random = MOTObjDetect(
            data_path, get_transform(False, data_augmentation), split_seqs=train_split_seqs)

    else:
        logger.error(
            "Please, provide a valid ds_train_name as argument. Select one of the following: motsynth_train, MOT17.")
        raise ValueError(ds_train_name)

    if ds_val_name == "motsynth_val":
        data_path = osp.join(
            MOTSYNTH_ROOT, 'comb_annotations', f"{ds_val_name}.json")
        dataset_val = get_mot_dataset(MOTSYNTH_ROOT,
                                      data_path, transforms=get_transform(False, data_augmentation))

    elif ds_val_name == "MOT17":
        test_split_seqs = ['MOT17-09-FRCNN']
        data_path = osp.join(MOTCHA_ROOT, "MOT17", "train")
        dataset_val = MOTObjDetect(data_path, get_transform(
            False, data_augmentation), split_seqs=test_split_seqs)

    else:
        logger.error(
            "Please, provide a valid ds_val_name as argument. Select one of the following: motsynth_val, MOT17.")
        raise ValueError(ds_val_name)

    return dataset_train, dataset_train_no_random, dataset_val


def create_data_loaders(dataset_train, dataset_train_no_random, dataset_test, batch_size, workers, aspect_ratio_group_factor):
    # random sampling on training dataset
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    # sequential sampling on test dataset
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset_train, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    # create train data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_batch_sampler, num_workers=workers, collate_fn=utils.collate_fn
    )

    data_loader_train_no_random = torch.utils.data.DataLoader(
        dataset_train_no_random, batch_size=1, sampler=test_sampler, num_workers=workers, collate_fn=utils.collate_fn
    )
    # create test data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=workers, collate_fn=utils.collate_fn
    )
    return data_loader_train, data_loader_train_no_random, data_loader_test


def create_optimizer(model, lr, momentum, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(
        f"Total trainable params: {total_trainable_params:}, Total params: {total_params}, lr: {lr}, momentum: {momentum}, weight_decay:{weight_decay}")

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


def resume_training(model, optimizer, lr_scheduler, scaler, resume, start_epoch, amp):
    checkpoint = torch.load(args.resume, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    args.start_epoch = checkpoint["epoch"] + 1
    if args.amp:
        scaler.load_state_dict(checkpoint["scaler"])


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


def show_sample(data_loader, model, device):
    for imgs, target in data_loader:
        with torch.no_grad():
            prediction = model([imgs[0].to(device)])[0]
        show_bbox(imgs[0], prediction, 0.75)
        show_bbox(imgs[0], target[0]['boxes'])
        break


def save_model_summary(model, output_dir, batch_size):
    with open(osp.join(output_dir, "summary.txt"), 'w', encoding="utf-8") as f:
        print(summary(model,
                      # (batch_size, color_channels, height, width)
                      input_size=(batch_size, 3, 1920, 1080),
                      verbose=0,
                      col_names=["input_size", "output_size",
                                 "num_params", "kernel_size", "trainable"],
                      col_width=20,
                      row_settings=["var_names"]), file=f)


def main(args):

    output_dir = None
    if args.output_dir:
        output_dir = osp.join(
            OUTPUT_DIR, 'detection_logs', args.output_dir)
        utils.mkdir(output_dir)
    output_plots_dir = osp.join(output_dir, "plots")
    utils.mkdir(output_plots_dir)

    device = torch.device(args.device)
    logger.debug(f"DEVICE: {device}")

    logger.debug("CREATE DATASETS")
    ds_train_name = args.train_dataset
    ds_val_name = args.val_dataset
    data_augmentation = args.data_augmentation
    dataset_train, dataset_train_no_random, dataset_test = create_datasets(
        ds_train_name, ds_val_name, data_augmentation)

    logger.debug("CREATE DATA LOADERS")
    batch_size = args.batch_size
    workers = args.workers
    aspect_ratio_group_factor = args.aspect_ratio_group_factor
    data_loader_train, data_loader_train_no_random, data_loader_test = create_data_loaders(
        dataset_train, dataset_train_no_random, dataset_test, batch_size, workers, aspect_ratio_group_factor)

    logger.debug("CREATE MODEL")
    model_name = args.model
    weights = args.weights
    backbone = args.backbone
    backbone_weights = args.backbone_weights
    trainable_backbone_layers = args.trainable_backbone_layers
    model = ModelFactory.get_model(
        model_name, weights, backbone, backbone_weights, trainable_backbone_layers)
    save_model_summary(model, output_dir, batch_size)
    model.to(device)

    if args.test_only:
        logger.debug("TEST ONLY")
        evaluate(model, data_loader_test, device=device, iou_types=['bbox'])
        show_sample(data_loader_train_no_random, model, device)
        return

    logger.debug("CREATE OPTIMIZER")
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    optimizer = create_optimizer(
        model, lr, momentum, weight_decay)

    logger.debug("CREATE LR SCHEDULER")
    epochs = args.epochs
    lr_scheduler_type = args.lr_scheduler.lower()
    lr_steps = args.lr_steps
    lr_gamma = args.lr_gamma
    lr_scheduler = create_lr_scheduler(
        optimizer, lr_scheduler_type, lr_steps, lr_gamma, epochs)

    logger.debug("CONFIGURE SCALER FOR amp")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        logger.debug("RESUME TRAINING")
        resume_training(model, optimizer, lr_scheduler,
                        scaler, args)

    logger.debug("START TRAINING")
    print_freq = args.print_freq
    start_epoch = args.start_epoch
    losses_dict = {}
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        _, batch_loss_dict = train_one_epoch(model, optimizer, data_loader_train, device,
                                             epoch, print_freq, scaler)
        lr_scheduler.step()
        save_plots(losses_dict, batch_loss_dict,
                   output_dir=output_plots_dir)

        # save model
        if epoch % 2 == 0:
            save_model_checkpoint(
                model, optimizer, lr_scheduler, epoch, scaler, output_dir, args)
            evaluate(model, data_loader_test,
                     device=device, iou_types=['bbox'])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug(f"TRAINING TIME: {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    logger.debug("COMMAND LINE ARGUMENTS")
    logger.debug(args)
    main(args)
