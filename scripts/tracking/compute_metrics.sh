#!/bin/sh

python core/tracking/TrackEval/scripts/run_mot_challenge.py --BENCHMARK CVCS --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL Frcnn_ByteTrack --METRICS CLEAR --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --PLOT_CURVES True