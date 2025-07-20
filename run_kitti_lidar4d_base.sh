#! /bin/bash

#if no input args use path = data/kitti360 else use input arg
if [ $# -eq 0 ]
then
    path="data/kitti360"
else
    path="$1/data/kitti360"
fi

CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_1538.txt \
--workspace log/kitti360_lidar4d_1538_debug6 \
--path $path \
--lr 1e-2 \
--num_rays_lidar 1024 \
--iters 13000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
--z_offsets -0.5 0.0 \
--fov_lidar 2.0 13.45 -11.45 13.45 \
#--ckpt scratch \

#--z_offsets -0.202 -0.121 \
#--num_layers_flow 2 \
#--hidden_dim_flow 32 \
#--num_frames 10 \
#--flow_loss False \
#--test_eval
# --refine
# --test_eval
