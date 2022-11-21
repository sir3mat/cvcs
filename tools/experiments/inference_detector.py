import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from configs.path_cfg import OUTPUT_DIR
from src.detection.graph_utils import add_bbox, plot_img_tensor
import os.path as osp
import argparse
from PIL import Image
import torchvision.transforms as transforms


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(
        description="Detector inference", add_help=add_help)

    # path to model used for inference
    parser.add_argument("--model-path", type=str,
                        help="Path to model checkpoint")

    # path of image used for evaluation
    parser.add_argument("--input", type=str,
                        help="Path to input image")

    parser.add_argument(
        '--threshold', default=0.75, type=float,
        help='detection threshold'
    )

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(
        args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    image = Image.open(args.input).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)[0]
    plot_img_tensor(add_bbox(image[0], prediction, args.threshold))


if __name__ == "__main__":
    args = parse_args()
    main(args)
