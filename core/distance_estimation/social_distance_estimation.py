import mouse_click_event
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from PIL import Image
from scipy.spatial import distance as dist
from settings import get_settings
settings = get_settings()


""" Preprocessing video frames """


def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocessed_image = frame.resize((settings["height"], settings["width"]))
    preprocessed_image = np.array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image


""" Utility function : Set the points in required order"""


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    x = points[np.argsort(points, 0)[:, 0]]
    x1 = x[:2, :]
    x2 = x[2:, :]
    rect[2] = x1[np.argsort(x1, 0)[:, 1]][1, :]
    rect[1] = x2[np.argsort(x2, 0)[:, 1]][1, :]
    rect[0] = x2[np.argsort(x2, 0)[:, 1]][0, :]
    rect[3] = x1[np.argsort(x1, 0)[:, 1]][0, :]
    return rect


""" Utility function : Get the transformation matrix """


def getmap(image):
    global grid_H, grid_W

    # image : first frame of the input video
    h, w = image.shape[:2]

    # 4 corner points of image are set by default
    corners = [[int(w*0.4), int(w*0.4)], [int(w*0.6), int(w*0.4)],
               [int(w*0.6), int(w*0.6)], [int(w*0.4), int(w*0.6)]]

    # User needs to finalise the corner points of a road or floor by dragging and dropping the corners
    corners = mouse_click_event.adjust_coor_quad(image, corners)
    corners = np.array(corners, dtype="float32")
    src = order_points(corners)

    (tl, tr, br, bl) = src
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(width1), int(width2))
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(height1), int(height2))
    width = int(width)
    height = int(height)
    dest = np.array([[0, 0], [width-1, 0], [width-1, height-1],
                    [0, height-1]], dtype="float32")
    M = cv.getPerspectiveTransform(src, dest)
    corners = np.array([[0, h-1, 1], [w-1, h-1, 1],
                       [w-1, 0, 1], [0, 0, 1]], dtype="int")
    warped_corners = np.dot(corners, M.T)
    warped_corners = warped_corners / \
        warped_corners[:, 2].reshape((len(corners), 1))
    warped_corners = np.int64(warped_corners)
    warped_corners = warped_corners[:, :2]
    min_coor = np.min(warped_corners, axis=0)
    max_coor = np.max(warped_corners, axis=0)
    grid_W, grid_H = max_coor-min_coor
    dest = np.array([[abs(min_coor[0]), abs(min_coor[1])], [abs(min_coor[0])+width-1, abs(min_coor[1])], [abs(
        min_coor[0])+width-1, abs(min_coor[1])+height-1], [abs(min_coor[0]), abs(min_coor[1])+height-1]], dtype="float32")
    M = cv.getPerspectiveTransform(src, dest)
    return M


""" Returns calibrated positions of detected people """


def calibration(M, results):

    # calculate minimum distance for social distancing
    global MIN_DISTANCE
    rect = np.array([r[1] for r in results])
    h = np.median(rect[:, 2]-rect[:, 0])
    coor = np.array([[50, 100, 1], [50, 100+h, 1]], dtype="int")
    coor = np.dot(coor, M.T)
    coor = coor/coor[:, 2].reshape((2, 1))
    coor = np.int64(coor[:, :2])
    MIN_DISTANCE = int(round(dist.pdist(coor)[0]))

    # calculate centroid points of detected people location corresponding to bird's eye view black grid
    centroids = np.array([r[2] for r in results])
    centroids = np.c_[centroids, np.ones((centroids.shape[0], 1), dtype="int")]
    warped_centroids = np.dot(centroids, M.T)
    warped_centroids = warped_centroids / \
        warped_centroids[:, 2].reshape((len(centroids), 1))
    warped_centroids = np.int64(warped_centroids)
    return warped_centroids[:, :2]


""" Returns a 2D numpy matrix consisting of list of pairs of indices indicating the positions of violators """


def calc_dist(centroids):

    # centroids : updated centroids in top-down view coordinates
    if len(centroids) < 2:  # no pair of people, no violation
        return list()

    # evaluate the pairwise distances between people
    condensed_dist = dist.pdist(centroids)
    D = dist.squareform(condensed_dist)
    locations = np.where(D < settings["min_distance"])
    violate = list(zip(locations[0], locations[1]))
    violate = np.sort(violate, axis=1)
    violate = np.unique(violate, axis=0)
    violate = np.asarray(list(filter(lambda x: x[0] != x[1], violate)))
    return violate


""" Visualise Output Frame """


def visualise_main(frame, results, violate):
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        colour = (0, 255, 0)
        if i in np.unique(violate):
            colour = (0, 0, 255)
        frame = cv.rectangle(frame, (startX, startY), (endX, endY), colour, 2)
        frame = cv.circle(frame, (cX, cY), 5, colour, 1)
    # Draw lines between violators
    for i, j in violate:
        frame = cv.line(frame, tuple(results[i][2]), tuple(
            results[j][2]), (0, 0, 255), 2)
    return frame


""" Visualise Bird's eye view grid and Warped Frames """


def visualise_grid(image, M, centroids, violate):
    warped = cv.warpPerspective(image, M, (grid_W, grid_H), cv.INTER_AREA,
                                borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    grid = np.zeros(warped.shape, dtype=np.uint8)
    for i in range(len(centroids)):
        colour = (0, 255, 0)  # green
        if i in np.unique(violate):
            colour = (0, 0, 255)  # red
        grid = cv.circle(grid, tuple(centroids[i, :]), 5, colour, -1)
        warped = cv.circle(warped, tuple(centroids[i, :]), 5, colour, -1)
    for i, j in violate:
        grid = cv.line(grid, tuple(centroids[i]), tuple(
            centroids[j]), (0, 0, 255), 2)
    grid = cv.resize(grid, image.shape[:2][::-1])
    warped = cv.resize(warped, image.shape[:2][::-1])
    return grid, warped


""" Pedestrian Detection """


def pedestrian_detector(image, model):
    H, W = settings["height"], settings["width"]  # changes
    img = np.asarray(image)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img[0])
    with torch.no_grad():
        output = model([img])
    detections = output[0]['boxes']
    scores = output[0]['scores']
    centroids = []
    boxes, confidences = [], []
    for i, score in enumerate(scores):
        if score > settings["confidence"]:
            box = detections[i].tolist()
            left, top = int(box[0]), int(box[1])
            right, bottom = int(box[2]), int(box[3])
            centerX = left + int((right - left) / 2)
            centerY = top + int((bottom - top) / 2)
            boxes.append([left, top, right - left, bottom - top])
            confidences.append(float(score))
            centroids.append((centerX, centerY))
    results = []
    if len(boxes) > 0:
        idxs = cv.dnn.NMSBoxes(
            boxes, confidences, settings["confidence"], settings["nms_threshold"])
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                results.append(
                    (confidences[i], (x, y, x+w, y+h), centroids[i]))
    return results
