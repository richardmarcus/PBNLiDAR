#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
source $HOME/.bashrc
#module load cuda/11.5.1
module load gcc/11.2.0
#module load python
source activate lidar4d

readonly JOB_CLASS="4D"

# $STAGING_DIR: place shared data there
readonly STAGING_DIR="/tmp/$USER-$JOB_CLASS-$9/"

# create staging directory, abort if it fails
(umask 0077; mkdir -p "$STAGING_DIR") || { echo "ERROR: creating $STAGING_DIR failed"; exit 1; }

# only one job is allowed to stage data, if others run at the same time they
# have to wait to avoid a race
echo "Begin staging section"
(
  exec {FD}>"$STAGING_DIR/.lock"
  flock "$FD"
  echo "Enter staging section"
  # check if another job has staged data already
  if [ ! -f "$STAGING_DIR/.complete" ]; then
    # START OF STAGING
    echo "Staging data to $STAGING_DIR"
    # -------------------------------------------------------
    # TODO: place here the code to copy data to $STAGING_DIR
    # -------------------------------------------------------
    cp -r "/home/atuin/b204dc/b204dc10/neural_rendering/nerf/LiDAR4D/data" "$STAGING_DIR/data"
    tar -xf "/home/atuin/b204dc/b204dc10/neural_rendering/nerf/LiDAR4D/train_$9.tar" -C "$STAGING_DIR"


    # END OF STAGING
    : > "$STAGING_DIR/.complete"
  fi
)
printf "Staging done\n"

# BELOW THIS LINE DATA STAGED TO $STAGING_DIR CAN BE USED.

#sequence="$1"
#setting="$2"
#tag="$3"
#opt_params="$4"
#lr_factors="$5"
#rolling_shutter="$6"
#lidar_dim="$7"
#reflectance_target="$8"
#path="$9/data/kitti360"

#run run_kitti_lidar4d.sh and give staging dir as datadir
./run_kitti_lidar4d.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$STAGING_DIR" "$9"

mkdir -p "/home/atuin/b204dc/b204dc10/neural_rendering/nerf/LiDAR4D/cluster_log"
cp -r "$STAGING_DIR/log/kitti360_lidar4d_f"$1"_"$9"_"$2"" "/home/atuin/b204dc/b204dc10/neural_rendering/nerf/LiDAR4D/cluster_log"

#copy new/updated data to $WORK before it gets deleted
#mkdir -p "/home/vault/b204dc/b204dc11/output/kitti_models/$cfg_file/$folder"
#cp -r "$STAGING_DIR/output/kitti_models/$cfg_file/$folder/$extra_tag" "/home/vault/b204dc/b204dc11/output/kitti_models/$cfg_file/$folder"

