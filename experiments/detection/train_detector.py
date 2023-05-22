from configs.path_cfg import OUTPUT_DIR
import datetime
import os.path as osp
import os
import time
import coloredlogs
import logging
import torch
import torch.utils.data
from core.detection.model_factory import ModelFactory
import core.detection.vision.utils as utils
from core.detection.vision.engine import train_one_epoch, evaluate
from experiments.detection.dataset_utils import create_dataset, get_transform, create_data_loader
from experiments.detection.training_utils import create_lr_scheduler, create_optimizer, save_args, save_evaluate_summary, save_model_checkpoint, save_model_summary, save_plots

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
    parser.add_argument("--train-dataset", default="motsynth_split3",
                        type=str, help="Dataset name. Please select one of the following:  motsynth_split1, motsynth_split2, motsynth_split3, MOT17 (default: motsynth_split3)")
    parser.add_argument("--val-dataset", default="MOT17",
                        type=str, help="Dataset name. Please select one of the following: MOT17 (default: MOT17)")

    # Transforms params
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="Data augmentation policy (default: hflip)"
    )

    # Data Loaders params
    parser.add_argument(
        "-b", "--batch-size", default=5, type=int, help="Images per gpu (default: 5)"
    )
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="Number of data loading workers"
    )
    parser.add_argument("--aspect-ratio-group-factor", default=-1,
                        type=int, help="Aspect ration group factor (default:disabled)")

    # Model param
    parser.add_argument(
        "--model", default="fasterrcnn_resnet50_fpn", type=str, help="Model name (default: fasterrcnn_resnet50_fpn)")
    parser.add_argument(
        "--weights", default="None", type=str, help="Model weights (default: None)"
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

    # Optimizer params
    parser.add_argument(
        "--lr",
        default=0.025,
        type=float,
        help="Learning rate (default: 0.025)",
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
        default=[8, 16, 22],
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

    # training param
    parser.add_argument("--start_epoch", default=0,
                        type=int, help="start epoch")
    parser.add_argument("--epochs", default=10, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument("--print-freq", default=20,
                        type=int, help="print frequency")

    return parser


def main(args):

    output_dir = None
    if args.output_dir:
        output_dir = osp.join(
            OUTPUT_DIR, 'detection_logs', args.output_dir)
        utils.mkdir(output_dir)
    output_plots_dir = osp.join(output_dir, "plots")
    utils.mkdir(output_plots_dir)

    logger.debug("COMMAND LINE ARGUMENTS")
    logger.debug(args)
    save_args(output_dir, args)

    device = torch.device(args.device)
    logger.debug(f"DEVICE: {device}")

    logger.debug("CREATE DATASETS")
    ds_train_name = args.train_dataset
    ds_val_name = args.val_dataset
    data_augmentation = args.data_augmentation

    dataset_train = create_dataset(
        ds_train_name, get_transform(True, data_augmentation), "train")
    dataset_test = create_dataset(
        ds_val_name, get_transform(False, data_augmentation), "test")

    logger.debug("CREATE DATA LOADERS")
    batch_size = args.batch_size
    workers = args.workers
    aspect_ratio_group_factor = args.aspect_ratio_group_factor
    data_loader_train = create_data_loader(
        dataset_train, "train", batch_size, workers, aspect_ratio_group_factor)
    data_loader_test = create_data_loader(
        dataset_test, "test", batch_size, workers)

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
        if (epoch % 5 == 0):
            save_model_checkpoint(
                model, optimizer, lr_scheduler, epoch, scaler, output_dir, args)
        coco_evaluator = evaluate(model, data_loader_test,
                                  device=device, iou_types=['bbox'])
        save_evaluate_summary(
            coco_evaluator.coco_eval['bbox'].stats, output_dir)

    save_model_checkpoint(
        model, optimizer, lr_scheduler, epoch, scaler, output_dir, args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug(f"TRAINING TIME: {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
