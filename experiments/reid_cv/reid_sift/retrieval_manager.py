import pickle
import cv2
import os.path as osp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalManager():

    def __init__(self, pkl_path, img_dir, k=100):
        self.pkl_path = pkl_path
        self.img_dir = img_dir

        self.descriptors = []
        with open(self.pkl_path, 'rb') as f:
            self.descriptors = pickle.load(f)
            

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, _, self.centres = cv2.kmeans(np.float32(np.vstack(self.descriptors)), k, None, criteria, 10, flags)

        hists = []
        for des in self.descriptors:
            hist = self.histogram(des, self.centres)
            hists.append(hist)

        hists = np.vstack(hists)
        self.occurrences = np.count_nonzero(hists, axis=0)

    def histogram(self, des, dict):
        dist = cdist(des, dict, 'euclidean')
        min_dist = np.argmin(dist, axis=1)

        hist = np.zeros(dict.shape[0])
        for idx in min_dist:
            hist[idx] += 1
        
        return hist

    def reweight(self, des, dict, idf):
        hist = self.histogram(des, dict)
        hist = hist / np.sum(hist)
        hist = hist * idf
        return hist

    def calc_des(self, name):
        path = osp.join(self.img_dir, name + '.png')
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            des = np.zeros((1,128))

        return des

    def retrieval(self, query_image, gallery, topk=5): 

        idf = np.log(len(self.descriptors) / np.hstack(self.occurrences))
        gallery_hists = []
        for name in gallery:
            des = self.calc_des(name)
            hist = self.reweight(des, self.centres, idf)
            gallery_hists.append(hist)

        des = self.calc_des(query_image)
        query_hist = self.reweight(des, self.centres, idf)

        similar_images = 1 - cosine_similarity(query_hist.reshape(1,-1), np.vstack(gallery_hists))
        similar_images = np.argsort(similar_images)

        return np.squeeze(similar_images[:,:topk])