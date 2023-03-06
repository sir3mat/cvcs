from typing import List
from configs.path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT
import os.path as osp
import coloredlogs
import logging
import torch
import torch.utils.data
from core.detection.vision.mot_data import MOTObjDetect
import core.detection.vision.presets as presets
import core.detection.vision.utils as utils
from core.detection.vision.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from core.detection.mot_dataset import get_mot_dataset
coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)

def get_transform(train, data_augmentation=None):
    if train:
        return presets.DetectionPresetTrain(data_augmentation)
    else:
        return presets.DetectionPresetEval()


def get_motsynth_dataset(ds_name: str, transforms):
    data_path = osp.join(MOTSYNTH_ROOT, 'comb_annotations', f"{ds_name}.json")
    dataset = get_mot_dataset(MOTSYNTH_ROOT, data_path, transforms=transforms)
    return dataset


def get_MOT17_dataset(split: str, split_seqs: List, transforms):
    data_path = osp.join(MOTCHA_ROOT, "MOT17", "train")
    dataset = MOTObjDetect(
        data_path, transforms=transforms, split_seqs=split_seqs)
    return dataset


def create_dataset(ds_name: str, transforms, split=None):
    if (ds_name.startswith("motsynth")):
        return get_motsynth_dataset(ds_name, transforms)

    elif (ds_name.startswith("MOT17")):
        if split == "train":
            split_seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN',
                          'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
        elif split == "test":
            split_seqs = ['MOT17-09-FRCNN', 'MOT17-10-FRCNN']
        return get_MOT17_dataset(split, split_seqs, transforms)

    else:
        logger.error(
            "Please, provide a valid dataset as argument. Select one of the following:  motsynth_split1, motsynth_split2, motsynth_split3, MOT17.")
        raise ValueError(ds_name)


def create_data_loader(dataset, split: str, batch_size, workers, aspect_ratio_group_factor=-1):
    data_loader = None
    if split == "train":
        # random sampling on training dataset
        train_sampler = torch.utils.data.RandomSampler(dataset)
        if aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(
                dataset, k=aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(
                train_sampler, group_ids, batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=workers, collate_fn=utils.collate_fn
        )
    elif split == "test":
        # sequential sampling on eval dataset
        test_sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=test_sampler, num_workers=workers, collate_fn=utils.collate_fn
        )
    return data_loader
