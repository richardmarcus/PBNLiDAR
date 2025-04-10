#! /bin/bash


#if no input args use path = data/kitti360 else use input arg
if [ $# -eq 0 ]
then
    path="data/kitti360"
    sequence="2350"
    setting="rolling_shutter"
    tag="RTopt_fresh"
    lidar_dim="2"
    rolling_shutter=""
    opt_params="R T"
    lr_factors="0.01 0.01"
    reflectance_target=0.0
    exit 0
else
    sequence="$1"
    setting="$2"
    tag="$3"
    opt_params="$4"
    lr_factors="$5"
    rolling_shutter="$6"
    lidar_dim="$7"
    reflectance_target="$8"
    path="$9""data/kitti360"
    scene=${10}
fi

flow="True"
#flow=""

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

echo "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10"

CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_"$sequence"_"$scene".txt \
--workspace "$9"log/kitti360_lidar4d_f"$sequence"_"$scene"_"$setting" \
--experiment_name $tag \
--path $path \
--lr 1e-2 \
--num_rays_lidar 1024 \
--iters 3000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
--fov_lidar $FOV_LIDAR \
--z_offsets $Z_OFFSETS \
--laser_offsets $laser_offsets \
--eval_interval 200 \
--out_lidar_dim $lidar_dim \
--motion "$rolling_shutter" \
--opt_params $opt_params \
--lr_factors $lr_factors \
--flow_loss "$flow" \
--reflectance_target $reflectance_target \
#--test_eval
#--refine \
#--refine \
#--ckpt scratch \
#--test_eval \
#--ckpt scratch \
#--test_eval \
#--refine \
#--test_eval \
#--ckpt scratch \
#--flow_loss True \
#--z_offsets -0.5 0.0 \ 
#--fov_lidar 2.0 13.45 -11.45 13.45 \
#--num_layers_flow 2 \
#--hidden_dim_flow 32 \
#--num_frames 10 \
#--flow_loss False \
#--test_eval
# --refine
# 
