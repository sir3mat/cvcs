from argparse import ArgumentParser
import os.path as osp
import torchreid
import torch 

from dataset import get_class


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--csv-mots', help="Path to motsynth csv file")
    parser.add_argument('--csv-mot17', help="Path to mot17 csv file")
    parser.add_argument('--img-mots', help='Directory where motsynth reid images are saved')
    parser.add_argument('--img-mot17', help='Directory where mot17 reid images are saved')
    
    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    dataset_names = ['MOT17', 'MOTSynth']
    for name in dataset_names:
        if 'MOT17' in name:
            dataset_class = get_class(name, args.csv_mot17, args.img_mot17)
        else:
            dataset_class = get_class(name, args.csv_mots, args.img_mots)
        torchreid.data.register_image_dataset(name, dataset_class)


    datamanager = torchreid.data.ImageDataManager(
        sources='MOTSynth',
        targets='MOT17'
    )

    model = torchreid.models.build_model(
        name="resnet18_fc512",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False
    )

    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )
    
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
    
    engine.run(
        save_dir="log/resnet18",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

    return

if __name__ == '__main__':
    main()