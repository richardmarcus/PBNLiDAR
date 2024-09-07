#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d_sim.py \
--config configs/kitti360_4950.txt \
--workspace log/kitti360_lidar4d_f4950_release/simulation \
--ckpt log/kitti360_lidar4d_f4950_release/checkpoints/lidar4d_ep0639_refine.pth \
--fov_lidar 2.0 26.9 \
--H_lidar 64 \
--W_lidar 1024 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0 \
--shift_z_bottom -.012 \
--align_axis \
#--kitti2nus
