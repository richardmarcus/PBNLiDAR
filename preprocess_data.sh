#! /bin/bash
DATASET="kitti360"
SEQ_ID="1538"
#FOV_LIDAR="2.0 13.45 -11.45 13.45"
#Z_OFFSETS="-0.5 0.0"
FOV_LIDAR="2.02984126984 11.0317460317 -8.799812 16.541"
Z_OFFSETS="-0.202 -0.121"

python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_id $SEQ_ID --fov_lidar $FOV_LIDAR --z_offsets $Z_OFFSETS

python -m data.preprocess.kitti360_to_nerf --sequence_id $SEQ_ID

python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID --fov_lidar $FOV_LIDAR --z_offsets $Z_OFFSETS