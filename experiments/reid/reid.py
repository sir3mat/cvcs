from argparse import ArgumentParser
import os.path as osp
import torchreid
import torch 

from dataset import get_class


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--csv-mots', help='Path to MOTSynth csv file')
    parser.add_argument('--img-mots', help='Directory where MOTSynth reid images are saved')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    dataset_class = get_class('MOTSynth', args.csv_mots, args.img_mots)
    torchreid.data.register_image_dataset('MOTSynth', dataset_class)


    datamanager = torchreid.data.ImageDataManager(
        sources='MOTSynth',
        targets=['market1501'],
        batch_size_train=64,
        batch_size_test=64
    )

    model = torchreid.models.build_model(
        name="resnet18_fc512",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    if torch.cuda.is_available():
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
        label_smooth=True
    )

    engine.run(
        save_dir="log/resnet18_fc512",
        max_epoch=30,
        fixbase_epoch=7,
        open_layers=['fc','classifier'],
        eval_freq=1,
        start_eval=1,
        test_only=False
    )

    return

if __name__ == '__main__':
    main()

