# Distance violation detector

This module aim to detect people in images and videos and measures the distance between them to identify potential social distancing violations.
This module uses the Detector module for people detection.

## Usage

For image, run:

```
python core/distance_estimation/image.py --input input.png --model path-to-model.pth
```

For video, run:

```
python core/distance_estimation/video.py --input input.mp4 --model path-to-model.pth
```


## Acknowledgements
- The original [DeepFusionAI/social-distance-detector](https://github.com/DeepFusionAI/social-distance-detector) repository that uses Tensorflow and YoloV3
