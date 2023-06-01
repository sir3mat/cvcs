import os
import sys
import cv2
import os.path as osp
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
    parser.add_argument('--ann-dir', type=str, required=True, help='Path to directory containing the mot17 json annotation files.')
    parser.add_argument('--imgs-dir', type=str, required=True, help='Directory where reid images are saved.')
    parser.add_argument('--pkl-dir', type=str, help='Directory where descriptors file is saved. If not provided, the annotation directory is used.')

    args = parser.parse_args()

    if args.pkl_dir is None:
        args.pkl_dir = osp.join(osp.dirname(osp.dirname(args.ann_dir)), 'MOTSynth')

    return args

def main(args):
    
    anns_file = [f for f in os.listdir(args.ann_dir) if 'train' not in f]
    mean_AP = []

    out = open('output.txt', 'a')

    for f in anns_file:
        ann_path = osp.join(args.ann_dir, f)
        print(ann_path)
        
        print('Generating query and gallery splits...')
    
        rh = RetrievalHelper(ann_path)
        query, gallery = rh.generate_retrieval_sets()

        print('Generating dictionary...')
        rm = RetrievalManager(osp.join(args.pkl_dir, 'descriptors.pkl'), args.imgs_dir)
        pt = Plotter()
    
        seq = ann_path.split('/')[-1][:-5]
        fldr = osp.join(osp.dirname(osp.dirname(args.imgs_dir)), 'motsynth_output/reid_sift_' + seq)
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
        mean = np.mean(AP_vector)
        print(mean)
        out.write(seq + ': ' + str(mean) + '\n')
        mean_AP.append(mean)

    print('computing mAP on all test datasets')
    print(np.mean(mean_AP))
    out.write('Total: ' + str(np.mean(mean_AP)) + '\n')
    
    out.close()

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)