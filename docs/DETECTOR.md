# Pedestrian detection

The Pedestrian detection module utilizes a Faster R-CNN architecture that has been specifically trained using synthetic data from MOTSynth. This training approach includes domain adaptation techniques to enhance the module's performance in real-world scenarios.

## Object Detection

An adaption of torchvision's detection reference code is done to train Faster R-CNN on a portion of the MOTSynth dataset.

- To train the model you can run (change params in the script):

```
./scripts/detector/train_detector.sh
```

- To fine-tuning the model you can run (change params in the script):

```
./scripts/detector/fine_tuning_detector.sh
```

- To evaluate the model you can run (change params in the script):

```
./scripts/detector/evaluate_detector.sh
```

- To make inference and show results you can run (change params in the script):

```
./scripts/detector/inference_detector.sh
```
