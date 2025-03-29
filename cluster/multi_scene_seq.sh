#!/bin/bash -l
#1538 1728 1908 3353
SEQ_IDs=(2350 4950 8120 10200 10750 11400)

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

