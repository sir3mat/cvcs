import os.path as osp
import argparse
from core.distance_estimation.social_distance_estimation import calibration, getmap, pedestrian_detector, preprocess_frame, visualise_grid, visualise_main, calc_dist
import torch
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from core.distance_estimation.settings import get_settings
settings = get_settings()


def process(image_path, output_path, model_path):

    model: models.detection.FasterRCNN = models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    checkpoint = torch.load(
        model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = cv.imread(image_path)
    image = cv.resize(image, (settings["width"], settings["height"]))

    """ Get the transformation matrix from the first frame """
    mat = getmap(image)

    print("Processing...")

    """ pedestrian detection """
    preprocessed_frame = preprocess_frame(image)
    results = pedestrian_detector(
        preprocessed_frame, model)
    preprocessed_frame = np.squeeze(preprocessed_frame) * 255.0
    preprocessed_frame = preprocessed_frame.clip(0, 255)
    preprocessed_frame = preprocessed_frame.squeeze()
    image = np.uint8(preprocessed_frame)

    """ calibration """
    warped_centroids = calibration(mat, results)

    """ Distance-Violation Determination """
    violate = calc_dist(warped_centroids)

    """ Visualise grid """
    grid, warped = visualise_grid(image, mat, warped_centroids, violate)

    """ Visualise main frame """
    image = visualise_main(image, results, violate)

    saveReportImage(image, grid, warped, output_path)
    print("Done!")


def saveReportImage(image, grid, warped, output_path):
    plt.figure(figsize=([120, 40]))
    plt.subplot(131), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title('Image', fontdict={'fontsize': 100}
              ), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(cv.cvtColor(warped, cv.COLOR_BGR2RGB))
    plt.title("Warped", fontdict={'fontsize': 100}
              ), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv.cvtColor(grid, cv.COLOR_BGR2RGB))
    plt.title("Bird's eye view grid", fontdict={
              'fontsize': 100}), plt.xticks([]), plt.yticks([])

    # save figure
    plt.savefig(fname=osp.join(output_path, "output.png"), format="png")
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image path',
                        default="street.png")
    parser.add_argument('--model', help='model path')
    parser.add_argument('--output', help='output path')

    args = parser.parse_args()
    process(args.input, args.output, args.model)
