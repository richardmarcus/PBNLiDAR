#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --export=NONE

printf("hi")