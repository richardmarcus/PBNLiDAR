#! /bin/bash



FOV_LIDAR="1.9647572 11.0334425 -8.979475  16.52717"
Z_OFFSETS="-0.20287499 -0.12243641"

laser_offsets=" 0.0101472   0.02935141 -0.04524597  0.04477938 -0.00623795  0.04855699 
-0.02581356 -0.00632023  0.00133613  0.05607248  0.00494516  0.00062785
0.03141189  0.02682017  0.01036519  0.02891498 -0.01124913  0.04208804
-0.0218643   0.00743873 -0.01018788 -0.01669445  0.00017374  0.0048293
0.03166919  0.03558188  0.01552001 -0.03950449  0.00887087  0.04522041
-0.04557779  0.01275884  0.02858396  0.06113308  0.03508026 -0.07183428
-0.10038704  0.02749107  0.0291795  -0.03833354 -0.07382096 -0.14437623
-0.09460489 -0.0584761   0.01881664 -0.02696179 -0.02052307 -0.15732896
-0.03719316 -0.00687183  0.07373429  0.03398049  0.04429062 -0.05352834
-0.07988049 -0.02726229 -0.00934669  0.09552395  0.0850026  -0.00946006
-0.05684165  0.0798225   0.10324192  0.08222152"

sequence_id="1538"
setting="combined_imask_nmask_2"
scene_id="0000"
log_path="log"
CUDA_VISIBLE_DEVICES=0 python main_pbl_sim.py \
--config configs/kitti360_$sequence_id"_"$scene_id.txt \
--workspace $log_path/kitti360_lidar4d_f$sequence_id"_"$scene_id"_"$setting/simulation \
--ckpt $log_path/kitti360_lidar4d_f$sequence_id"_"$scene_id"_"$setting/checkpoints/big_improved_ep0400_refine.pth \
--H_lidar 64 \
--W_lidar 1024 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--align_axis \
--interpolation_factor 0.0 \
--out_lidar_dim 3 \
--fov_lidar $FOV_LIDAR \
--laser_offsets $laser_offsets \
--shift_z_top -0.20287499  \
--shift_z_bottom -0.12243641 \


