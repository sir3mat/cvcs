import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from configs.path_cfg import OUTPUT_DIR
from src.detection.vision.engine import evaluate
from tools.train_detector import create_dataset, create_data_loader, get_transform
from src.detection.graph_utils import add_bbox, show_img
import os.path as osp
import argparse


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(
        description="Detector inference", add_help=add_help)

    # path to model used for inference
    parser.add_argument("--model-path", type=str,
                        help="Path with model checkpoint used for inference")

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = osp.join(
            OUTPUT_DIR, "detection_logs", "fasterrcnn_training", "checkpoint.pth")
    return args


def main(args):
    ds_val = create_dataset(
        "motsynth_val", get_transform(False, "hflip"), "test")
    data_loader_val = create_data_loader(ds_val, "test", 1, 0)

    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn_v2()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(
        args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    show_img(data_loader_val, model, device, 0.8)


if __name__ == "__main__":
    args = parse_args()
    main(args)
