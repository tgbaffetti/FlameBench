#!/usr/bin/env bash
#SBATCH --job-name=flamebench
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=flamebench-%A_%a.log
set -euo pipefail
# Submit from repository root; activate the cluster's Python environment beforehand.
models=(gru lstm transformer)
python -m Experiments.run run --config "Experiments/Configs/pod_${models[$SLURM_ARRAY_TASK_ID]}.json" --seeds 0 1 2
