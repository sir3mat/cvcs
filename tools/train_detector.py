from configs.path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT, OUTPUT_DIR
import datetime
import os.path as osp
import os
import time
import coloredlogs
import logging
import torch
import torch.utils.data

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights
from src.detection.utils import save_train_loss_plot

import src.detection.vision.presets as presets
import src.detection.vision.utils as utils
from src.detection.vision.engine import train_one_epoch, evaluate
from src.detection.vision.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from src.detection.mot_dataset import get_mot_dataset

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Detection Training", add_help=add_help)

    # train and validtion dataset params
    parser.add_argument("--train-dataset", default="motsynth",
                        type=str, help="dataset name")
    parser.add_argument("--val-dataset", default="MOT17",
                        type=str, help="dataset name")

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
    parser.add_argument("--epochs", default=30, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 0)"
    )
    parser.add_argument(
        "--lr",
        default=0.005,
        type=float,
        help="initial learning rate, 0.005 is the default value",
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


def get_dataset(name, data_path, transform):
    if name == 'motsynth':
        ann_file = osp.join(data_path, 'comb_annotations', 'train.json')

    elif name == 'MOT17':
        ann_file = osp.join(
            data_path, 'motcha_coco_annotations', 'MOT17_train.json')

    else:
        logger.error(
            "Please, provide a valid dataset as argument. Select one of the following: motsynth, MOT17_train.")
        raise ValueError(name)

    logger.debug(
        f"get_dataset -> ann_file: {ann_file}")

    ds = get_mot_dataset(data_path, ann_file, transforms=transform)

    return ds, 2


def create_dataset(name: str, data_path: str, transform):
    logger.debug(
        f"create_dataset -> name: {name}, data path: {data_path}")
    dataset, num_classes = get_dataset(
        name, data_path, transform)
    return dataset, num_classes


def resume_training(model, optimizer, lr_scheduler, scaler, args):
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    args.start_epoch = checkpoint["epoch"] + 1
    if args.amp:
        scaler.load_state_dict(checkpoint["scaler"])


class ModelFactory:
    @staticmethod
    def get_model(args):
        if args.model == "fasterrcnn_resnet50_fpn":
            backbone_name = "resnet50"
            trainable_backbone_layers = 5
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            # added for baseline
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2
            model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(weights=weights, backbone_name=backbone_name, weights_backbone=backbone_weights,
                                                           trainable_backbone_layers=trainable_backbone_layers)
            num_classes = 2  # 1 class (person) + background
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)
        else:
            logger.error(
                "Please, provide a valid model as argument. Select one of the following: fasterrcnn.")
            raise ValueError(args.model)
        return model


def save_model_checkpoint(model, optimizer, lr_scheduler, epoch, scaler, args):
    if args.output_dir:
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
            args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(
            args.output_dir, "checkpoint.pth"))


def main(args):
    # create ouput_dir
    if args.output_dir:
        args.output_dir = osp.join(
            OUTPUT_DIR, 'detection_logs', args.output_dir)
        utils.mkdir(args.output_dir)

    logger.info("COMMAND LINE ARGUMENTS")
    logger.info(args)

    device = torch.device(args.device)

    logger.debug("CREATING DATASETS")
    dataset_train, num_classes = create_dataset(
        args.train_dataset, MOTSYNTH_ROOT, get_transform(True, args))
    dataset_test, _ = create_dataset(
        args.val_dataset, MOTCHA_ROOT, get_transform(False, args))
    logger.debug(
        f"train_dataset:{dataset_train}, num_classes:{num_classes}, validation_dataset:{dataset_test}")

    logger.debug("CREATING DATA LOADERS")
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
    # create test data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    logger.debug("CREATING MODEL")
    logger.debug(f"MODEL: {args.model}")
    model = ModelFactory.get_model(args)
    model.to(device)

    # perform only test
    if args.test_only:
        evaluate(model, data_loader_test, device=device, iou_types=['bbox'])
        return

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(
        f"training params: {total_trainable_params:}, total params: {total_params}")

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # construct a learning rate scheduler
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    # configure scaler for automatic mixed precision training (amp)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # resume training
    if args.resume:
        resume_training(model, optimizer, lr_scheduler,
                        scaler, args)

    logger.debug("START TRAINING")

    losses_dict = {}

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        _, batch_loss_dict = train_one_epoch(model, optimizer, data_loader_train, device,
                                             epoch, args.print_freq, scaler)
        lr_scheduler.step()
        evaluate(model, data_loader_test,
                 device=device, iou_types=['bbox'])

        # save train loss plot
        if not losses_dict:
            for name, metric in batch_loss_dict.items():
                losses_dict[name] = []
        for name, metric in batch_loss_dict.items():
            losses_dict[name].extend(metric)
        save_train_loss_plot(losses_dict, OUTPUT_DIR)

        # save model
        if (epoch % 2 == 0):
            save_model_checkpoint(
                model, optimizer, lr_scheduler, epoch, scaler, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug(f"TRAINING TIME: {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
