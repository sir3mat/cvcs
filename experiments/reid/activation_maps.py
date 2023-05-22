from torchreid.utils import (
    check_isfile, mkdir_if_missing, load_pretrained_weights
)
from torch.nn import functional as F
import cv2
import numpy as np
from torchreid.utils.model_complexity import compute_model_complexity
from torchreid.utils.tools import set_random_seed
from core.reid.mot_reid_dataset import get_sequence_class
from core.reid.deep_person_reid.torchreid.data.datasets import __image_datasets
from experiments.reid.default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
)
import torchreid
import os
import torch.nn as nn
import torch
import argparse
from PIL import Image
from configs.path_cfg import OUTPUT_DIR
import os.path as osp
Image.MAX_IMAGE_PIXELS = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
    model,
    test_loader,
    save_dir,
    width,
    height,
    use_gpu,
    img_mean=None,
    img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]['query']  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, 'actmap_' + target)
        mkdir_if_missing(actmap_dir)
        print('Visualizing activation maps for {} ...'.format(target))

        for batch_idx, data in enumerate(data_loader):
            imgs, paths = data['img'], data['impath']
            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                outputs = model(imgs, return_featuremaps=True)
            except TypeError:
                raise TypeError(
                    'forward() got unexpected keyword argument "return_featuremaps". '
                    'Please add return_featuremaps as an input argument to forward(). When '
                    'return_featuremaps=True, return feature maps only.'
                )

            if outputs.dim() != 4:
                raise ValueError(
                    'The model output is supposed to have '
                    'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                    'Please make sure you set the model output at eval mode '
                    'to be the last convolutional feature maps'.format(
                        outputs.dim()
                    )
                )

            # compute activation maps
            outputs = (outputs**2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            if use_gpu:
                imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                # activation map
                am = outputs[j, ...].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
                )
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # overlapped
                overlapped = img_np*0.3 + am*0.7
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones(
                    (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
                )
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:,
                         width + GRID_SPACING:2*width + GRID_SPACING, :] = am
                grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

            if (batch_idx+1) % 10 == 0:
                print(
                    '- done batch {}/{}'.format(
                        batch_idx + 1, len(data_loader)
                    )
                )


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def update_datasplits(cfg):
    def get_split_seqs(name):
        splits = ['motsynth_split1', 'motsynth_split2', 'motsynth_split3',
                  'motsynth_split4', 'motsynth_train', 'motsynth', 'motsynth_val']
        motsynth_splits = [
            f'{split}{maybe_mini}' for split in splits for maybe_mini in ('_mini', '')]
        assert name in ['market1501', 'cuhk03', 'mot17'] + \
            motsynth_splits, f"Got dataset name {name}"
        if name in ('market1501', 'cuhk03', 'motsynth_train'):
            return name

        if name == 'mot17':
            return [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 5, 9, 10, 11, 13)]

        return name

    assert isinstance(cfg.data.sources, (tuple, list))
    assert isinstance(cfg.data.sources, (tuple, list))
    cfg.data.sources = [get_split_seqs(ds_name)
                        for ds_name in cfg.data.sources]
    cfg.data.targets = [get_split_seqs(ds_name)
                        for ds_name in cfg.data.targets]

    if isinstance(cfg.data.sources[0], (tuple, list)) and len(cfg.data.sources) == 1:
        cfg.data.sources = cfg.data.sources[0]

    if isinstance(cfg.data.targets[0], (tuple, list)) and len(cfg.data.targets) == 1:
        cfg.data.targets = cfg.data.targets[0]


def register_datasets(cfg):
    for maybe_data_list in (cfg.data.sources, cfg.data.targets):
        if not isinstance(maybe_data_list, (tuple, list)):
            maybe_data_list = [maybe_data_list]

        for seq_name in maybe_data_list:
            #print("Registering dataset ", seq_name)
            if seq_name not in __image_datasets:
                seq_class = get_sequence_class(seq_name)
                torchreid.data.register_image_dataset(seq_name, seq_class)


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    update_datasplits(cfg)
    register_datasets(cfg)

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    test_loader = datamanager.test_loader

    visactmap(
        model, test_loader, cfg.data.save_dir, cfg.data.width, cfg.data.height, cfg.use_gpu
    )


if __name__ == '__main__':
    main()
