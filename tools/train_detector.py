from configs.path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT, OUTPUT_DIR
import datetime
import os.path as osp
import os
import time
import coloredlogs
import logging
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

    # train and validtion dataset params
    parser.add_argument("--train-dataset", default="motsynth_train",
                        type=str, help="dataset name. Please select one of the following: motsynth_train, MOT17")
    parser.add_argument("--val-dataset", default="motsynth_val",
                        type=str, help="dataset name. Please select one of the following: motsynth_val, MOT17")

    # model param
    parser.add_argument(
        "--model", default="fasterrcnn_resnet50_fpn", type=str, help="model name")

    # device param
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")

    # training param
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)"
    )
    parser.add_argument(
        "--lr",
        default=0.0025,
        type=float,
        help="initial learning rate, 0.0025 is the default value",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--momentum", default=0.9,
                        type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    parser.add_argument(
        "--trainable-backbone-layers", default=3, type=int, help="number of trainable layers of backbone (default is 3)"
    )
    parser.add_argument(
        "--backbone", default='resnet50', type=str, help="ResNet backbone used"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )

    # resume params
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")

    # test mode param
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # pretrained param
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
        default=True
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")

    # other param
    parser.add_argument("--print-freq", default=20,
                        type=int, help="print frequency")
    parser.add_argument("--output-dir", default='fasterrcnn_training',
                        type=str, help="path to save outputs")

    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int)
    parser.add_argument("--rpn-score-thresh", default=None,
                        type=float, help="rpn score threshold for faster-rcnn")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    return parser


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(args.data_augmentation)
    else:
        return presets.DetectionPresetEval()


def create_datasets(ds_train_name, ds_val_name, args):
    dataset_train = dataset_train_no_random = dataset_val = None

    if ds_train_name == "motsynth_train":
        data_path = osp.join(
            MOTSYNTH_ROOT, 'comb_annotations', f"{ds_train_name}.json")
        dataset_train = get_mot_dataset(
            data_path, transforms=get_transform(True, args))
        dataset_train_no_random = get_mot_dataset(
            data_path, transforms=get_transform(False, args))

    elif ds_train_name == "MOT17":
        train_split_seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
                            'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
        data_path = osp.join(MOTCHA_ROOT, "MOT17", "train")
        dataset_train = MOTObjDetect(data_path, get_transform(
            True, args), split_seqs=train_split_seqs)
        dataset_train_no_random = MOTObjDetect(
            data_path, get_transform(False, args), split_seqs=train_split_seqs)

    else:
        logger.error(
            "Please, provide a valid ds_train_name as argument. Select one of the following: motsynth_train, MOT17.")
        raise ValueError(ds_train_name)

    if ds_val_name == "motsynth_val":
        data_path = osp.join(
            MOTSYNTH_ROOT, 'comb_annotations', f"{ds_val_name}.json")
        dataset_val = get_mot_dataset(
            data_path, transforms=get_transform(False, args))

    elif ds_val_name == "MOT17":
        test_split_seqs = ['MOT17-09-FRCNN']
        dataset_test = MOTObjDetect(data_path, get_transform(
            False, args), split_seqs=test_split_seqs)

    else:
        logger.error(
            "Please, provide a valid ds_val_name as argument. Select one of the following: motsynth_val, MOT17.")
        raise ValueError(ds_val_name)
    
    return dataset_train, dataset_train_no_random, dataset_test


def create_data_loaders(dataset_train, dataset_train_no_random, dataset_test, args):
    # random sampling on training dataset
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    # sequential sampling on test dataset
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset_train, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    # create train data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_train_no_random = torch.utils.data.DataLoader(
        dataset_train_no_random, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    # create test data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
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


def create_lr_scheduler(optimizer, args):
    lr_scheduler_type = args.lr_scheduler.lower()
    if lr_scheduler_type == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        logger.debug(
            f"lr_scheduler: {lr_scheduler_type}, milestones: {args.lr_steps}, gamma: {args.lr_gamma}")

    elif lr_scheduler_type == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
        logger.debug(
            f"lr_scheduler: {lr_scheduler_type}, T_max: {args.epochs}")
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler_type}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    return lr_scheduler


def resume_training(model, optimizer, lr_scheduler, scaler, args):
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


def save_plot(losses_dict, batch_loss_dict, output_dir):
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


def main(args):
    output_dir = None
    if args.output_dir:
        output_dir = osp.join(
            OUTPUT_DIR, 'detection_logs', args.output_dir)
        utils.mkdir(output_dir)

    logger.debug("COMMAND LINE ARGUMENTS")
    logger.debug(args)

    device = torch.device(args.device)
    logger.debug(f"DEVICE: {device}")

    logger.debug("CREATE DATASETS")
    dataset_train, dataset_train_no_random, dataset_test = create_datasets(
        args.train_dataset, args.val_dataset)

    logger.debug("CREATE DATA LOADERS")
    data_loader_train, data_loader_train_no_random, data_loader_test = create_data_loaders(
        dataset_train, dataset_train_no_random, dataset_test, args)

    logger.debug("CREATE MODEL")
    model = ModelFactory.get_model(args)
    model.to(device)

    if args.test_only:
        logger.debug("TEST ONLY")
        evaluate(model, data_loader_test, device=device, iou_types=['bbox'])
        show_sample(data_loader_train_no_random, model, device)
        return

    logger.debug("CREATE OPTIMIZER")
    optimizer = create_optimizer(
        model, args.lr, args.momentum, args.weight_decay)

    logger.debug("CREATE LR SCHEDULER")
    lr_scheduler = create_lr_scheduler(optimizer, args)

    logger.debug("CONFIGURE SCALER FOR amp")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        logger.debug("RESUME TRAINING")
        resume_training(model, optimizer, lr_scheduler,
                        scaler, args)

    logger.debug("START TRAINING")
    losses_dict = {}
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        _, batch_loss_dict = train_one_epoch(model, optimizer, data_loader_train, device,
                                             epoch, args.print_freq, scaler)
        lr_scheduler.step()
        # save train loss plot
        save_plot(losses_dict, batch_loss_dict, output_dir)

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
    main(args)
