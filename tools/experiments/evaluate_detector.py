from configs.path_cfg import  OUTPUT_DIR
import os.path as osp
import coloredlogs
import logging
import torch
import torch.utils.data
import src.detection.vision.utils as utils
from src.detection.vision.engine import  evaluate
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tools.experiments.dataset_utils import create_dataset, get_transform, create_data_loader
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
    parser.add_argument("--val-dataset", default="MOT17",
                        type=str, help="Dataset name. Please select one of the following: MOT17 (default: MOT17)")


    # Data Loaders params
    parser.add_argument(
        "-b", "--batch-size", default=5, type=int, help="Images per gpu (default: 5)"
    )
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="Number of data loading workers"
    )
    # Device param
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (default: cuda)")

    # Path to model used for evaluation
    parser.add_argument(
        "--model-eval", type=str, help="model path"
    )
    return parser

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
    ds_val_name = args.val_dataset
    dataset_test = create_dataset(
        ds_val_name, get_transform(False), "test")

    logger.debug("CREATE DATA LOADERS")
    batch_size = args.batch_size
    workers = args.workers
    data_loader_test = create_data_loader(
    dataset_test, "test", batch_size, workers)

    logger.debug("TEST ONLY")
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(args.model_eval, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    coco_evaluator = evaluate(model, data_loader_test,
                                device=device, iou_types=['bbox'])
    save_evaluate_summary(
        coco_evaluator.coco_eval['bbox'].stats, output_dir)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
