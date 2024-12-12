#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d_sim.py \
--config configs/kitti360_1538.txt \
--workspace log/kitti360_lidar4d_f1538_optimize/simulation \
--ckpt log/kitti360_lidar4d_f1538_optimize/checkpoints/lidar4d_ep0334_refine.pth \
--H_lidar 64 \
--W_lidar 1024 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--align_axis \
--interpolation_factor 0.0 \
--shift_z_top -0.202 \
--shift_z_bottom -0.121 \
--fov_lidar 2.02984126984 11.0317460317 -8.799812 16.541 \
# -0.382 \
#--shift_z_bottom -0.121 \
#--fov_lidar 2.0 13.45 -11.45 13.45 \
#--num_layers_flow 2 \
#--hidden_dim_flow 32 \
# --kitti2nus
#--workspace log/kitti360_lidar4d_f1538_release/simulation \
#--ckpt log/kitti360_lidar4d_f1538_release/checkpoints/lidar4d_ep0500_refine.pth \


