#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d_sim.py \
--config configs/kitti360_3353.txt \
--workspace log/kitti360_lidar4d_f3353_release/simulation \
--ckpt log/kitti360_lidar4d_f3353_release/checkpoints/lidar4d_ep0500_refine.pth \
--fov_lidar 2.02984126984 11.0317460317 -8.799812 16.541 \
--H_lidar 256 \
--W_lidar 4096 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--shift_z_top -0.202 \
--shift_z_bottom -0.121 \
--align_axis \
--interpolation_factor 0.75 \
#--num_layers_flow 2 \
#--hidden_dim_flow 32 \
# --kitti2nus
