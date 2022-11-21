import logging
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import ResNet50_Weights
import torch
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


class ModelFactory:
    @staticmethod
    def get_model(name, weights, backbone, backbone_weights, trainable_backbone_layers):
        logger.debug(f"get_model -> model:{name}")

        if name == "fasterrcnn_resnet50_fpn":
            if weights == "None":
                model_weights = None
            else:
                model_weights = FasterRCNN_ResNet50_FPN_Weights[weights]
            model_backbone_weights = ResNet50_Weights[backbone_weights]
            model: FasterRCNN = fasterrcnn_resnet50_fpn(
                weights=model_weights, backbone_name=backbone, weights_backbone=model_backbone_weights, trainable_backbone_layers=trainable_backbone_layers)

            set_seeds()

            num_classes = 2  # 1 class (person) + background
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

        else:
            logger.error(
                "Please, provide a valid model as argument. Select one of the following: fasterrcnn_resnet50_fpn.")
            raise ValueError(name)

        return model
