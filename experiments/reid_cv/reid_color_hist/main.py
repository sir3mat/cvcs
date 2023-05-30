import cv2
import os.path as osp
import os
import sys
import numpy as np
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retrieval_manager import RetrievalManager
from retrieval_helper import RetrievalHelper
from evaluation.plotter import Plotter
from evaluation.evaluator import Evaluator



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ann-path', type=str, required=True, help='Path to the mot17 json annotation file.')
    parser.add_argument('--imgs-dir', type=str, required=True, help='Directory where reid images are saved.')

    args = parser.parse_args()

    return args



def main(args):
    print('Generating query and gallery splits...')

    rh = RetrievalHelper(args.ann_path)
    query, gallery = rh.generate_retrieval_sets()
    
    print('Initializing...')
    rm = RetrievalManager(args.imgs_dir)
    
    pt = Plotter()
    
    if 'MOTSynth' in args.imgs_dir:
        fldr = osp.join(osp.dirname(osp.dirname(args.imgs_dir)), 'motsynth_output/reid_color_hist_mots')
    else:
        fldr = osp.join(osp.dirname(osp.dirname(args.imgs_dir)), 'motsynth_output/reid_color_hist_mot17')
    
    if not osp.exists(fldr):
        os.mkdir(fldr)

    AP_vector = []
    query = query.reset_index().drop('index', axis=1)
    for i, row in query.iterrows():
        print('Processing image ' + str(i+1) + ' of ' + str(len(query)) + ' (' + str(row['id']) + ')...')

        similar_images = rm.retrieval(row['id'], gallery=gallery['id'].values, topk=10)

        query_image = cv2.cvtColor(cv2.imread(osp.join(args.imgs_dir, str(row['id']) + '.png')), cv2.COLOR_BGR2RGB)
        results = []
        labels = []
        for idx in similar_images:
            img = cv2.cvtColor(cv2.imread(osp.join(args.imgs_dir, str(gallery['id'].values[idx]) + '.png')), cv2.COLOR_BGR2RGB)
            results.append(img)
            labels.append(gallery['ped_id'].values[idx])
        
        pt.plot_retrieval_results(fldr, i+1, len(results), query_image, results)
        
        print('starting evaluation...')
        img_evaluator = Evaluator(query_image, results)
        single_AP = img_evaluator.eval(row['ped_id'], labels)
        
        AP_vector.append(single_AP)
        print("AP vector:")
        print(AP_vector)
        print()

    print("computing MAP on test dataset ")
    print(np.mean(AP_vector))

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)