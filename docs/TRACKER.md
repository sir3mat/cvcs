# Pedestrian tracking

Pedestrian tracking module follows the tracking by detection paradigm using the pedestrian detection module and ByteTrack for track people

# How to use

- To generate prediction you can run (change params in the script):

```
./scripts/tracking/generate_preds.sh
```

- To generate an annotated video (change params in the script):

```
./scripts/tracking/generate_ann_video.sh
```

- To compute metrics you can run (change params in the script):

```
./scripts/tracking/compute_metrics.sh
```
