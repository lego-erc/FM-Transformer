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

# Outside a job: read run.nodes from the config and submit ourselves with it
# (a #SBATCH directive cannot be computed at runtime; the CLI flag overrides it).
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    CFG_PATH="$CONFIG"
    [[ "$CFG_PATH" == *.yaml ]] || CFG_PATH="$CFG_PATH.yaml"
    [[ -f "$CFG_PATH" ]] || CFG_PATH="configs/$CFG_PATH"
    NODES=$(awk '/^run:/{f=1; next} f && /^[^ ]/{exit} f && $1=="nodes:"{print $2; exit}' "$CFG_PATH")
    exec sbatch --nodes="${NODES:-1}" "$0" "$CONFIG"
fi

srun pixi run --frozen -e train-comet python scripts/train.py "$CONFIG"
