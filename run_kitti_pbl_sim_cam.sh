#! /bin/bash
sequence_id="1538"
setting="combined_imask_nmask_2"
scene_id="0000"
log_path="log"
CUDA_VISIBLE_DEVICES=0 python main_pbl_sim.py \
--config configs/kitti360_$sequence_id"_"$scene_id.txt \
--workspace $log_path/kitti360_lidar4d_f$sequence_id"_"$scene_id"_"$setting/simulation \
--ckpt $log_path/kitti360_lidar4d_f$sequence_id"_"$scene_id"_"$setting/checkpoints/big_improved_ep0400_refine.pth \
--H_lidar 376 \
--W_lidar 1408 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--align_axis \
--interpolation_factor 0.0 \
--use_camera \
--out_lidar_dim 3 \

