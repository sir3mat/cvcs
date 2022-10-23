import os.path as osp
import gradio as gr
import torch
import logging
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.utils import draw_bounding_boxes

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from configs.path_cfg import MOTCHA_ROOT, OUTPUT_DIR
from src.detection.vision import presets
logging.getLogger('PIL').setLevel(logging.CRITICAL)


def add_bbox(img, output, th=None):
    img_to_show = torch.clip(img*255, 0, 255)
    img_to_show = img_to_show.type(torch.uint8)
    img_with_bbbox = None
    if th == None:
        img_with_bbbox = draw_bounding_boxes(
            img_to_show, boxes=output, width=4)
    else:
        img_with_bbbox = draw_bounding_boxes(
            img_to_show, boxes=output["boxes"][output['scores'] > th], width=4)
    return img_with_bbbox


def load_model():
    model = fasterrcnn_resnet50_fpn_v2()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(osp.join(OUTPUT_DIR, "detection_logs",
                            "fasterrcnn_training", "checkpoint.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.eval()
    model.to("cuda")
    return model


def run_inference(image):
    model = load_model()
    transformEval = presets.DetectionPresetEval()
    image_tensor = transformEval(image, None)
    print(image_tensor)
    prediction = model(image_tensor)[0]
    image_w_bbox = add_bbox(image, prediction, 0.75)
    torchvision.io.write_png(image_w_bbox, "out.png")
    return "out.png"


title = "Performance comparision of Faster R-CNN for people detection with syntetic data"
description = "Performance comparision of Faster R-CNN models for people detecion using MOTSynth and MOT17"
examples = [[osp.join(MOTCHA_ROOT, "MOT17", "train",
                      "MOT17-09-DPM", "img1", "000001.jpg")]]
gr.Interface(run_inference, gr.Image(type="pil"), gr.Image(
    type="file"), title=title, description=description, examples=examples).launch(enable_queue=True)
