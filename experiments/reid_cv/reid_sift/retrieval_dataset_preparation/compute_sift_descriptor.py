from argparse import ArgumentParser
import json
import pickle
import cv2
import os.path as osp
import pandas as pd
from sift_helper import SIFTHelper
import tqdm
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ann-path', type=str, required=True, help='Path to the mot17 json annotation file.')
    parser.add_argument('--imgs-dir', type=str, required=True, help='Directory where reid images are saved.')
    parser.add_argument('--pkl-dir', type=str, help='Directory where descriptors file is saved. If not provided, the annotation directory is used.')
    parser.add_argument('--pb-path', type=str, help='Full path to pb model for super resolution.')

    args = parser.parse_args()

    if args.pkl_dir is None:
        args.pkl_dir = osp.dirname(osp.dirname(args.ann_path))

    return args


def main(args):
    dataset_path = args.imgs_dir
    descriptors_path = osp.join(args.pkl_dir, 'descriptors.pkl')

    data = json.load(open(args.ann_path))

    ped_ids = []
    ids = []
    for ann in data['annotations']:
        if ann['bbox'][-1] >= 50 and ann['bbox'][-2] >= 25:
            ped_ids.append(ann['ped_id'])
            ids.append(ann['id'])
    
    df = pd.DataFrame(columns=['id', 'ped_id'])
    df['id'] = ids
    df['ped_id'] = ped_ids
    tmp = df.groupby('ped_id')['id'].count()
    df = df.join(tmp, on='ped_id', how='left', rsuffix='_count')
    df = df[df['id_count'] >= 100]
    
    fn = lambda obj: obj.loc[np.random.choice(obj.index, 100, False),:]
    df = df.groupby('ped_id', as_index=False).apply(fn)

    #sh = SIFTHelper()

    descriptors = []
    
    for img in tqdm.tqdm(df['id'].values):
        path = osp.join(dataset_path, str(img) + '.png')
        
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        #img = sh.super_res(img, args.pb_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create()
        _, des = sift.detectAndCompute(gray_img, None)
        if des is None:
            des = np.zeros((1,128))

        descriptors.append(des)
        
    with open(descriptors_path, 'wb') as f:    
        pickle.dump(descriptors, f)

    f.close()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)