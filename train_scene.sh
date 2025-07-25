#!/bin/bash -l

# Configuration variables
SCENE_ID="0000"
SEQ_ID="1538"
TAG="default"
SETTING="combined_imask_nmask_2"
OPT_PARAMS="laser_strength near_range_threshold near_range_factor distance_scale near_offset distance_fall"
LR_FACTORS="0.1 0.05 0.05 0.005 0.1 0.1"
ROLLING_SHUTTER="True"
LIDAR_DIM=3
REFLECTANCE_TARGET=0.2

# Function to run KITTI PBL
run_kitti() {
    ./run_kitti_pbl.sh \
        "$SEQ_ID" \
        "$SETTING" \
        "$TAG" \
        "$OPT_PARAMS" \
        "$LR_FACTORS" \
        "$ROLLING_SHUTTER" \
        "$LIDAR_DIM" \
        "$REFLECTANCE_TARGET" \
        "" \
        "$SCENE_ID"
}

# Execute the script
run_kitti
