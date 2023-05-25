import json
import os.path as osp
import pandas as pd
import numpy as np



class RetrievalHelper():

    def __init__(self, anns_path):
        self.anns_path = anns_path
        self.data = json.load(open(anns_path))

    
    # generating train, query, gallery sets for retrieval
    def generate_retrieval_sets(self, count=15):
        ped_ids = []
        ids = []
        for img in self.data['annotations']:
            if img['bbox'][-1] >= 50 and img['bbox'][-2] >= 25:
                ped_ids.append(img['ped_id'])
                ids.append(img['id'])

        df = pd.DataFrame(columns=['ped_id', 'id'])
        df['ped_id'] = ped_ids
        df['id'] = ids
        tmp = df.groupby('ped_id')['id'].count()
        df = df.join(tmp, on='ped_id', how='left', rsuffix='_count')
        df = df[df['id_count'] >= count]

        df['index'] = df.index.values
        np.random.seed(0)
        query_per_id = df.groupby('ped_id')['index'].agg(lambda x: np.random.choice(list(x.unique())))
        query = df.loc[query_per_id.values].copy()
        query.drop(['index'], axis=1, inplace=True)

        gallery = df.drop(query_per_id).copy()
        gallery.drop(['index'], axis=1, inplace=True)

        return query, gallery