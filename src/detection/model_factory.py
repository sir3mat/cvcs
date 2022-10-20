import logging
import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchinfo import summary

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


class ModelFactory:
    @staticmethod
    def get_model(args):
        logger.debug(f"get_model -> model:{args.model}")

        if args.model == "fasterrcnn_resnet50_fpn":
            backbone_name = "resnet50"
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            trainable_backbone_layers = 1
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2
            model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(
                weights=weights, backbone_name=backbone_name, weights_backbone=backbone_weights, trainable_backbone_layers=trainable_backbone_layers)
            num_classes = 2  # 1 class (person) + background
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

        elif args.model == "frcnn_custom":
            backbone_name = "resnet50"
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            trainable_backbone_layers = 0
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2
            model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(
                weights=weights, backbone_name=backbone_name, weights_backbone=backbone_weights, trainable_backbone_layers=trainable_backbone_layers)

            for param in model.features.parameters():
                param.requires_grad = False
            set_seeds()

            num_classes = 2
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

        else:
            logger.error(
                "Please, provide a valid model as argument. Select one of the following: fasterrcnn.")
            raise ValueError(args.model)

        # # Get a summary of the model (uncomment for full output)
        summary(model,
                # (batch_size, color_channels, height, width)
                input_size=(args.batch_size, 3, 1920, 1080),
                verbose=0,
                col_names=["input_size", "output_size",
                           "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model
