import numpy as np
from .plotter import Plotter


class RetrievalMeasure():
    def __init__(self) -> None:
        self.pt = Plotter()
        pass

    # return: the AP value
    def get_AP(self, label_q, labels, rank):
        precisions = []
        num_cur_relevant = 0
        recalls = []
        for i in range(rank):
            if labels[i] == label_q:
                num_cur_relevant += 1 
                recalls.append(1)
            else:
                recalls.append(0)
            precisions.append(num_cur_relevant / (i + 1))
        
        return np.mean(np.array(precisions) * np.array(recalls))

    def compute_MAP(self, AP_vector):
        return np.mean(AP_vector)