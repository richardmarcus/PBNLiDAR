#!/bin/bash -l

SEQ_IDs="1538 1728 1908 3353 2350 4950 8120 10200 10750 11400"

#sequence="$1"
#setting="$2"
#tag="$3"
#opt_params="$4"
#lr_factors="$5"
#rolling_shutter="$6"
#lidar_dim="$7"
tag="default"

for SEQ_ID in $SEQ_IDs
do
  bash cluster/dcluster_train.sh $SEQ_ID "laser" $tag "laser_strength" "0.1" "" 2
  exit 0
done

