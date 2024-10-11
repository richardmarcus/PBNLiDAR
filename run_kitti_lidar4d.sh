#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_3353.txt \
--workspace log/kitti360_lidar4d_3353_release \
--lr 1e-2 \
--num_rays_lidar 1024 \
--iters 30000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
--z_offsets -0.202 -0.121 \
--fov_lidar 2.02984126984 11.0317460317 -8.799812 16.541 \
#--num_layers_flow 2 \
#--hidden_dim_flow 32 \
#--num_frames 10 \
#--flow_loss False \
#--test_eval
# --refine
# --test_eval
