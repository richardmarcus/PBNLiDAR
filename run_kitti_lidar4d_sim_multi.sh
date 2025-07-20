#!/bin/bash

# Define the common part of the base command (excluding config, workspace, interpolation_factor, and ckpt)
BASE_CMD="python main_lidar4d_sim.py \
--fov_lidar 2.02984126984 11.0317460317 -8.799812 16.541 \
--H_lidar 64 \
--W_lidar 1024 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--shift_z_top -0.202 \
--shift_z_bottom -0.121 \
--align_axis"

# List of configurations and their corresponding checkpoints
CONFIGS=("1538" "1908" "3353")
CHECKPOINTS=(
  "log/kitti360_lidar4d_f1538_release/checkpoints/lidar4d_ep0500_refine.pth"
  "log/kitti360_lidar4d_f1908_release/checkpoints/lidar4d_ep0500_refine.pth"
  "log/kitti360_lidar4d_f3353_release/checkpoints/lidar4d_ep0500_refine.pth"
)

# List of interpolation factors
FACTORS=(0.0 0.25 0.5 0.75)

# Loop through each configuration and factor
for i in "${!CONFIGS[@]}"; do
    config=${CONFIGS[$i]}
    ckpt=${CHECKPOINTS[$i]}

    for factor in "${FACTORS[@]}"; do
        # Construct workspace name based on config and factor
        workspace="log/kitti360_lidar4d_${config}_release/simulation${factor//./}"

        # Run the simulation with the current config, checkpoint, and interpolation factor
        echo "Running simulation with config ${config}, checkpoint ${ckpt}, and interpolation factor ${factor}"
        
        CUDA_VISIBLE_DEVICES=0 $BASE_CMD --config configs/kitti360_${config}.txt \
                  --workspace $workspace \
                  --ckpt $ckpt \
                  --interpolation_factor $factor
    done
done