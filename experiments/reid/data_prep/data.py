import json
import os
import os.path as osp
import pandas as pd
from argparse import ArgumentParser
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--anns-dir', type=str, required=True, help="Path containing the MOT17 json files with annotations")
    parser.add_argument('--imgs-dir', type=str, required=True, help="Path containing the MOT17 reid images")
    parser.add_argument('--csv-dir', type=str, required=True, help='Directory in which the generated csv annotations will be stored')

    args = parser.parse_args()

    return args.anns_dir, args.imgs_dir, args.csv_dir

def read_json(filepath):
    f = open(filepath)
    data = json.load(f)
    return data

def read_anns(data, train=False):
    if not train:
        df = pd.DataFrame(columns=['image', 'ped_id'])
        for row in data['annotations']:
            name = row['id'] + '.png'
            tmp = pd.DataFrame(data=[[name, row['ped_id']]], columns=['image', 'ped_id'])
            df = pd.concat([df, tmp], ignore_index=True)
    else:
        df = pd.DataFrame(columns=['image', 'ped_id', 'height', 'width'])
        for row in data['annotations']:
            name = row['id'] + '.png'
            tmp = pd.DataFrame(data=[[name, row['ped_id'], row['bbox'][-1], row['bbox'][-2]]], columns=['image', 'ped_id', 'height', 'width'])
            df = pd.concat([df, tmp], ignore_index=True)
    return df

def max_el(df, f):
    M = df['ped_id'].max()
    s = '[%s] Max element: %d' % (f[:8], M)
    print(s)

def min_el(df, f):
    m = df['ped_id'].min()
    s = '[%s] Min element: %d' % (f[:8], m)
    print(s)

def unique_id(df, f):
    ids = df['ped_id'].unique()
    s = '[%s] Unique ids: [' % f[:8]
    for id in ids:
        s = s + str(id) + ', '
    s = s[:-2]
    s += ']' 
    print(s)
    return ids

def len_id(ids, f):
    l = len(ids)
    s = '[%s] Len ids: %d' % (f[:8], l)
    print(s)

def check_exist(df, path_imgs):
    s = ''
    for index, row in df.iterrows():
        img = osp.join(path_imgs, row['image'])
        if not osp.exists(img):
            s = '[MOT%s-%s] Image %s does not exist.' % (row['image'][0:2], row['image'][2:4], row['image'])
            break
    if not s:
        s = '[MOT%s-%s] All images exist.' % (df.loc[0, 'image'][0:2], df.loc[0, 'image'][2:4])
    print(s)

def len_df(df, f):
    l = len(df)
    s = '[%s] Len df: %d' % (f[:8], l)
    print(s)

def csv_writer(df, path_imgs, f, csv_fldr, train=False):
    csv_name = '%s.csv' % f[:8]
    csv_writer = open(osp.join(csv_fldr, csv_name), 'w')
    if not train:
        csv_writer.write('image,ped_id\n')
        for index, row in df.iterrows():
            name = osp.join(path_imgs, row['image'])
            csv_writer.write('{},{}\n'.format(name, row['ped_id']))
    else:
        csv_writer.write('image,ped_id,height,width\n')
        for index, row in df.iterrows():
            csv_writer.write('{},{},{},{}\n'.format(row['image'], row['ped_id'], row['height'], row['width']))
    csv_writer.close()



if __name__ == '__main__':

    json_fldr, path_imgs, csv_fldr = parse_args()
    json_files = os.listdir(json_fldr)
    json_files = json_files[:-1]
    """ for f in json_files:
        data = read_json(osp.join(json_fldr, f))
        mot_df = read_anns(data)
        max_el(mot_df, f)
        min_el(mot_df, f)
        ids = unique_id(mot_df, f)
        len_id(ids, f)
        check_exist(mot_df, path_imgs)
        len_df(mot_df, f)
        print('Writing csv...')
        csv_writer(mot_df, path_imgs, f, csv_fldr)
        print('Done!')
        print()
 """
    data = read_json(osp.join(json_fldr, 'MOT17_train.json'))
    mot = read_anns(data, train=True)
    l = len(mot)
    s = '[MOT_train] Len df: %d' % l
    print(s)
    csv_writer(mot, path_imgs, 'MOT17_train.json', csv_fldr, True)

