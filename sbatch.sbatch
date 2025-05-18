#!/bin/sh
#!SBATCH -p epito
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -G 2
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH -t 00:10:00

srun -n 1  ./build/core/mpc ./launch.toml &
srun -n 2 --gpus-per-task=1  ./build/core/mpc ./launch.toml 
