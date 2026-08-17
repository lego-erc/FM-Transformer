#!/bin/bash -l
#SBATCH -o /ptmp/hildebra/lego_fm_models/job.out.%j
#SBATCH -e /ptmp/hildebra/lego_fm_models/job.err.%j
#SBATCH -D /u/hildebra/phd/legofmt/FM-Transformer
#SBATCH -J train_legofmt
#SBATCH --constraint="gpu"
#SBATCH --nvmps
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=none
#SBATCH --mail-user=richard.hildebrandt@tum.de
#SBATCH --time=4:59:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CONFIG="${1:-fm_default}"

srun pixi run --frozen -e train-comet python scripts/train.py "$CONFIG"
