from argparse import ArgumentParser
import os.path as osp
import torchreid
import torch

from experiments.reid.mot_reid_dataset import get_sequence_class


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--csv-mots', help='Path to MOTSynth csv file')
    parser.add_argument(
        '--img-mots', help='Directory where MOTSynth reid images are saved')
    parser.add_argument('--root', help='Path to Market1501 dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    dataset_class = get_sequence_class('motsynth_split3')
    torchreid.data.register_image_dataset('MOTSynth', dataset_class)

    datamanager = torchreid.data.ImageDataManager(
        sources='MOTSynth',
        targets='market1501',
        batch_size_train=64,
        batch_size_test=64,
        workers=0,
        root=args.root,
        use_gpu=torch.cuda.is_available(),
    )

    model = torchreid.models.build_model(
        name="resnet18_fc512",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
        use_gpu=torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(torch.device('cuda:0'))
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="amsgrad",
        lr=0.0009
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=15
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        use_gpu=torch.cuda.is_available()
    )

    engine.run(
        save_dir="log/resnet18_fc512",
        max_epoch=30,
        fixbase_epoch=3,
        open_layers=['fc', 'classifier'],
        eval_freq=1,
        start_eval=1,
        test_only=False
    )

    return


if __name__ == '__main__':
    main()
