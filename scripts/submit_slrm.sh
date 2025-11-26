#!/bin/bash -l
#SBATCH -J train_legofmt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=125000
#SBATCH --time=0-02:00:00

srun pixi run train