#!/bin/bash -l

SEQ_IDs=(1538 1728 1908 3353 2350 4950 8120 10200 10750 11400)

tag="default"

#SEQ_ID is first element
SEQ_ID=${SEQ_IDs[0]}
echo $SEQ_ID

sbatch --job-name=${SEQ_ID}_reflectance2 cluster/cluster_train.sh $SEQ_ID "reflectance" $tag "" "" "" 3 0.0

