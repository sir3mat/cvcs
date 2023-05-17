import numpy as np
import pandas as pd
from torchreid.data import ImageDataset
import os.path as osp


def clean_rows(df, min_h, min_w, min_samples):
    keep = (df['height'] >= min_h) & (df['width'] >= min_w)
    clean_df = df[keep]

    clean_df['samples_per_id'] = clean_df.groupby(
        'ped_id')['height'].transform('count').values
    clean_df = clean_df[clean_df['samples_per_id'] >= min_samples]

    return clean_df

def relabel_ids(df):
    df.rename(columns={'ped_id': 'ped_id_old'}, inplace=True)

    # Relabel Ids from 0 to N-1
    ids_df = df[['ped_id_old']].drop_duplicates()
    ids_df['ped_id'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)
    return df


class MOTDataset(ImageDataset):

    def __init__(self, dataset_dir, img_dir, min_samples, min_h=50, min_w=25, **kwargs):
        df = pd.read_csv(dataset_dir)
        df['image'] = img_dir + '/' + df['image']
        df = clean_rows(df, min_h, min_w, min_samples)
        df = relabel_ids(df)

        def to_tuple_list(df): return list(
            df[['image', 'ped_id', 'cam_id']].itertuples(index=False, name=None))
        df['cam_id'] = 0
        train = to_tuple_list(df)

        df['index'] = df.index.values
        query_per_id = df.groupby('ped_id')['index'].agg(
            lambda x: np.random.choice(list(x.unique())))
        query_df = df.loc[query_per_id.values].copy()
        gallery_df = df.drop(query_per_id).copy()
        gallery_df['cam_id'] = 1

        query = to_tuple_list(query_df)
        gallery = to_tuple_list(gallery_df)

        super(MOTDataset, self).__init__(train, query, gallery, **kwargs)


def get_class(name, dataset_dir, img_dir):

    if 'MOT17' in name:
        min_samples = 5

    else:
        min_samples = 15

    class MOT(MOTDataset):
        def __init__(self, **kwargs):
            super(MOT, self).__init__(dataset_dir=dataset_dir,
                                      img_dir=img_dir, min_samples=min_samples, **kwargs)

    MOT.__name__ = name

    return MOT
