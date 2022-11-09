import os.path as osp
from tkinter.ttk import Style
import gradio as gr
import torch
import logging
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from configs.path_cfg import MOTCHA_ROOT, OUTPUT_DIR
from src.detection.graph_utils import add_bbox
from src.detection.vision import presets
logging.getLogger('PIL').setLevel(logging.CRITICAL)


def load_model(baseline: bool = False):
    if baseline:
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        checkpoint = torch.load(osp.join(OUTPUT_DIR, "detection_logs",
                                "fasterrcnn_training", "checkpoint.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def detect_with_resnet50Model_finetuning_motsynth(image):
    model = load_model()
    transformEval = presets.DetectionPresetEval()
    image_tensor = transformEval(image, None)[0]
    prediction = model([image_tensor])[0]
    image_w_bbox = add_bbox(image_tensor, prediction, 0.85)
    torchvision.io.write_png(image_w_bbox, "custom_out.png")
    return "custom_out.png"


def detect_with_resnet50Model_baseline(image):
    model = load_model(baseline=True)
    transformEval = presets.DetectionPresetEval()
    image_tensor = transformEval(image, None)[0]
    prediction = model([image_tensor])[0]
    image_w_bbox = add_bbox(image_tensor, prediction, 0.85)
    torchvision.io.write_png(image_w_bbox, "baseline_out.png")
    return "baseline_out.png"


title = "Performance comparision of Faster R-CNN for people detection with syntetic data"
description = "<p style='text-align: center'>Performance comparision of Faster R-CNN models for people detecion using MOTSynth and MOT17"
examples = [[osp.join(MOTCHA_ROOT, "MOT17", "train",
                      "MOT17-09-DPM", "img1", "000001.jpg")]]


io_baseline = gr.Interface(detect_with_resnet50Model_baseline, gr.Image(type="pil"), gr.Image(
    type="file", shape=(1920, 1080), label="FasterR-CNN_Resnet50_COCO"))

io_custom = gr.Interface(detect_with_resnet50Model_finetuning_motsynth, gr.Image(type="pil"), gr.Image(
    type="file", shape=(1920, 1080), label="FasterR-CNN_Resnet50_FinteTuning_MOTSynth"))

gr.Parallel(io_baseline, io_custom, title=title,
            description=description, examples=examples).launch(enable_queue=True)
