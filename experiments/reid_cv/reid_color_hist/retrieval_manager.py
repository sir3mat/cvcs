import cv2
import os.path as osp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalManager():

    def __init__(self, img_dir):
        self.img_dir = img_dir


    def get_vector(self, image, bins=32):
        red = cv2.calcHist([image], [2], None, [bins], [0, 256])
        green = cv2.calcHist([image], [1], None, [bins], [0, 256])
        blue = cv2.calcHist([image], [0], None, [bins], [0, 256])

        vector = np.concatenate([red, green, blue], axis=0)
        
        vector = vector.reshape(1, -1)
        return vector

    def retrieval(self, query_image, gallery, topk=5): 
        
        gallery_hists = []

        for name in gallery:
            path = osp.join(self.img_dir, str(name) + '.png')
            img = cv2.imread(path)
            color_hist = self.get_vector(img)
            gallery_hists.append(color_hist)

        img = cv2.imread(osp.join(self.img_dir, str(query_image) + '.png'))
        query_hist = self.get_vector(img).reshape(1, -1)

        distances = []
        for _, vector in enumerate(gallery_hists):
            dist = cosine_similarity(query_hist, vector).reshape(-1)
            distances.append(dist[0])
        top_idx = np.argpartition(distances, -topk)[-topk:]

        return top_idx