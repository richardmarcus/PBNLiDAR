#! /bin/bash

#exit on errer
set -e

SEQ_IDs="1538" # 1728 1908 3353 2350 4950 8120 10200 10750 11400"


DATASET="kitti360"

for SEQ_ID in $SEQ_IDs
do
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

    python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_id $SEQ_ID --fov_lidar $FOV_LIDAR --z_offsets $Z_OFFSETS --laser_offsets $laser_offsets
  
    python -m data.preprocess.kitti360_to_nerf --sequence_id $SEQ_ID

    python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID --fov_lidar $FOV_LIDAR --z_offsets $Z_OFFSETS --laser_offsets $laser_offsets
done

python -m data.preprocess.generate_mask
python -m data.preprocess.generate_intensitymask

