import os
from loguru import logger
from core.tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from core.tracking.ByteTrack.yolox.motdt_tracker.motdt_tracker import STrack
import numpy as np
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from typing import List
from supervision.tools.detections import Detections
import os.path as osp
import coloredlogs
import logging
import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from experiments.detection.dataset_utils import  get_MOT17_dataset, get_transform, create_data_loader
coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)


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


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for item in results:
            x1 = item["x1"]
            y1 = item["y1"]
            w = item["w"]
            h = item["h"]
            score = item["score"]
            tracker_id = item["tracker_id"]
            frame = item["frame"]
            line = save_format.format(frame=frame, id=tracker_id, x1=int(
                x1), y1=int(y1), w=w, h=h, s=round(score, 2))
            f.write(line)
    logger.info('save results to {}'.format(filename))


def generate_prediction(results_folder: str, ts_file: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"DEVICE: {device}")

    logger.debug("CREATE DATASETS")
    split_seqs = [ts_file]
    dataset_test = get_MOT17_dataset("test", split_seqs, get_transform(False))

    logger.debug("CREATE DATA LOADERS")
    batch_size = 5
    workers = 0
    data_loader = create_data_loader(
        dataset_test, "test", batch_size, workers)

    model_path = "c:/Users/Matteo/Desktop/cvcs/storage/pretrained_models/model_split3_FT_MOT17.pth"
    device = torch.device("cuda")
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # class_ids of interest - person
    CLASS_ID = [1]

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    res = []
    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
        results = model(images)
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
            img_info=images[0].shape,
            img_size=images[0].shape
        )
        tracker_id = match_detections_with_tracks(
            detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        image_id = targets[0]["image_id"].numpy()
        path = dataset_test._img_paths[image_id[0]]
        img1, name = osp.split(path)
        frame = int(name.split(".")[0])
        for item in detections:
            obj = {"x1": item[0][0], "y1": item[0][1], "w": item[0][2] - item[0][0], "h": item[0]
                   [3] - item[0][1], "score": item[1], "tracker_id": item[-1], "frame": frame}
            res.append(obj)

    write_results(os.path.join(results_folder, ts_file +
                  ".txt").replace("\\", "/"), res)


if __name__ == "__main__":
    mot17_09 = "MOT17-09-FRCNN"
    mot17_10 = "MOT17-10-FRCNN"
    results_folder = 'c:/Users/Matteo/Desktop/cvcs/storage/mota'

    generate_prediction(results_folder, mot17_09)
    generate_prediction(results_folder, mot17_10)
