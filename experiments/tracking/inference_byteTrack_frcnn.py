from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
import torchvision.transforms as transforms
from tqdm import tqdm
import os.path as osp

import numpy as np
import torch

from core.tracking.ByteTrack.yolox.motdt_tracker.motdt_tracker import STrack
from core.tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Tracking inference", add_help=add_help)

    parser.add_argument("--source-video",
                        type=str, help="source video path")

    parser.add_argument("--target-video",
                        type=str, help="target video path")

    parser.add_argument("--model",
                        type=str, help="detection model path")

    return parser



@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# def preditct_annotate_single_frame():
#     SOURCE_VIDEO_PATH = "c:/Users/Matteo/Desktop/cvcs/082.mp4"

#     model_path = "c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"
#     device = torch.device("cuda")
#     model = fasterrcnn_resnet50_fpn()
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
#     checkpoint = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(checkpoint["model"])
#     model.to(device)
#     model.eval()

#     # dict maping class_id to class_name
#     CLASS_NAMES_DICT = {1: "person"}
#     # class_ids of interest - person
#     CLASS_ID = [1]

#     # create frame generator
#     generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
#     # create instance of BoxAnnotator
#     box_annotator = BoxAnnotator(
#         color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
#     # acquire first video frame
#     iterator = iter(generator)
#     frame = next(iterator)

#     # model prediction on single frame and conversion to supervision Detections
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     image = transform(frame).to(device)
#     image = image.unsqueeze(0)
#     results = model(image)

#     results = results[0]
#     boxes_t: torch.Tensor = results["boxes"]
#     scores_t: torch.Tensor = results["scores"]
#     labels_t: torch.Tensor = results["labels"]

#     detections = Detections(
#         xyxy=boxes_t.detach().cpu().resolve_conj().resolve_neg().numpy(),
#         confidence=scores_t.detach().cpu().resolve_conj().resolve_neg().numpy(),
#         class_id=labels_t.detach().cpu().resolve_conj().resolve_neg().numpy().astype(int)
#     )
#     # format custom labels
#     labels = [
#         f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, tracker_id
#         in detections
#     ]
#     # annotate and display frame
#     frame = box_annotator.annotate(
#         frame=frame, detections=detections, labels=labels)

#     show_frame_in_notebook(frame, (16, 16))


def preditct_annotate_video(source_video_path: str, target_video_path: str, model_path: str):
    SOURCE_VIDEO_PATH = source_video_path
    # settings
    LINE_START = Point(50, 1500)
    LINE_END = Point(3840-50, 1500)
    TARGET_VIDEO_PATH = target_video_path
    VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    model_path = model_path
    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = {1: "person"}
    # class_ids of interest - person
    CLASS_ID = [1]

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
   
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(
    ), thickness=1, text_thickness=1, text_scale=0.5)
    
    # open target video file
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
           # model prediction on single frame and conversion to supervision Detections
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(frame).to(device)
            image = image.unsqueeze(0)
            results = model(image)

            results = results[0]
            boxes_t: torch.Tensor = results["boxes"]
            scores_t: torch.Tensor = results["scores"]
            labels_t: torch.Tensor = results["labels"]

            detections = Detections(
                xyxy=boxes_t.detach().cpu().resolve_conj().resolve_neg().numpy(),
                confidence=scores_t.detach().cpu().resolve_conj().resolve_neg().numpy(),
                class_id=labels_t.detach().cpu().resolve_conj().resolve_neg().numpy().astype(int)
            )

            # filtering out detections with unwanted classes
            mask = np.array(
                [class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            # annotate and display frame
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels)
            sink.write_frame(frame)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    source_video_path = args.source_video
    target_video_path = args.target_video
    model_path = args.model
    preditct_annotate_video(source_video_path, target_video_path, model_path)
