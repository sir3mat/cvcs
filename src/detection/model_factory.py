import coloredlogs
import logging

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def get_model(args):
        if args.model == "fasterrcnn_resnet50_fpn":
            backbone_name = "resnet50"
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            trainable_backbone_layers = 1
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2
            model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(
                weights=weights, backbone_name=backbone_name, weights_backbone=backbone_weights, trainable_backbone_layers=trainable_backbone_layers)
            num_classes = 2  # 1 class (person) + background
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)
        else:
            logger.error(
                "Please, provide a valid model as argument. Select one of the following: fasterrcnn.")
            raise ValueError(args.model)
        logger.debug(f"get_model -> model:{args.model}")
        return model
