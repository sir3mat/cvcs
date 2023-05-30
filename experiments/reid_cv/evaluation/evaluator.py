from .evaluate_retrieval import RetrievalMeasure


class Evaluator():
    def __init__(self, query_img, results) -> None:
        self.query_img = query_img
        self.results = results
        self.rm = RetrievalMeasure()

    def eval(self, label_q, labels):

        single_AP = self.rm.get_AP(label_q, labels, len(labels))
        print("Average Precision: " + str(single_AP))
        return single_AP

    def compute_MAP(self, AP_test):
        self.rm.compute_MAP(AP_test)