#!/bin/bash -l
#1538 1728 1908 3353
SEQ_IDs=(250 500 750 1322 1572 1822 2072 2322 2572 2822 3072 3322 3572 3822 4072 4360 4610 4860 5110 5371 5932 6182 6432 6682 6932 7182 7432 7682 7932 8201 8451 8701 8951 9609 9859 10109 10375 10625 10875 11143)

#sequence="$1"
#setting="$2"
#tag="$3"
#opt_params="$4"
#lr_factors="$5"
#rolling_shutter="$6"
#lidar_dim="$7"
tag="debug_refine"

for SEQ_ID in ${SEQ_IDs[@]}
do
  #bash cluster/dcluster_train.sh $SEQ_ID "laser" $tag "laser_strength" "0.1" "" 2
  #cluster/dcluster_train.sh $SEQ_ID "poses" $tag "R T" "0.01 0.01" "" 2
  #cluster/dcluster_train.sh $SEQ_ID "distance" $tag "near_range_threshold near_range_factor distance_scale near_offset distance_fall" "0.05 0.05 0.01 0.1 0.1" "" 2 0.0
  #cluster/dcluster_train.sh $SEQ_ID "reflectance_nmask" $tag "" "" "" 3 0.2
  #cluster/dcluster_train.sh $SEQ_ID "imask" $tag "" "" "" 2 0.0
  cluster/dcluster_train.sh $SEQ_ID "combined_imask_nmask" $tag "laser_strength near_range_threshold near_range_factor distance_scale near_offset distance_fall" "0.1 0.05 0.05 0.005 0.1 0.1" "True" 3 0.2
  exit 0
done

