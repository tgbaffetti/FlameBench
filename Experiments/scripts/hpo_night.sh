#!/usr/bin/env bash
# Night HPO battery (2026-09-23): 11 latent/classical pairs, trials 25, seeds 0 1 2.
# One sequential lane per GPU plus a CPU lane. Operators excluded (Transolver smoke run
# still holds GPU 2 until morning; DeepONet HPO deferred).
#
# Launch DETACHED from the repository root (apptainer stays foreground of the detached
# shell — `nohup apptainer &` bus-errors on squashfuse unmount):
#   setsid nohup bash Experiments/scripts/hpo_night.sh > Experiments/Results/hpo_night.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/../.."
repo="$PWD"
sif="$HOME/miniconda.sif"
python="$HOME/envs/dtd2c/bin/python"

run() { # run <gpu|cpu> <config-stem>
    local gpu="$1" cfg="$2"
    local log="Experiments/Results/night_${cfg}.log"
    echo "[$(date +%FT%T)] START ${cfg} (gpu=${gpu})"
    if [ "$gpu" = "cpu" ]; then
        apptainer exec --bind /srv/mlg --env "PYTHONPATH=${repo}" --env CUDA_VISIBLE_DEVICES= \
            "$sif" "$python" -m Experiments.run hpo --config "Experiments/Configs/${cfg}.json" \
            --device cpu --seeds 0 1 2 > "$log" 2>&1
    else
        apptainer exec --nv --bind /srv/mlg --env "PYTHONPATH=${repo}" --env "CUDA_VISIBLE_DEVICES=${gpu}" \
            "$sif" "$python" -m Experiments.run hpo --config "Experiments/Configs/${cfg}.json" \
            --seeds 0 1 2 > "$log" 2>&1
    fi
    echo "[$(date +%FT%T)] DONE  ${cfg} exit=$? (log: ${log})"
}

lane() { local gpu="$1"; shift; for cfg in "$@"; do run "$gpu" "$cfg"; done; }

lane 0 hpo_cae_arx hpo_cae_gru hpo_cae_transformer &
lane0=$!
lane 1 hpo_pod_gru hpo_pod_cnn hpo_pod_lstm hpo_pod_transformer &
lane1=$!
lane 2 hpo_cae_cnn hpo_cae_lstm hpo_cae_narx &
lane2=$!
lane cpu hpo_pod_arx hpo_pod_narx &
cpu_lane=$!

wait "$lane0" "$lane1" "$lane2" "$cpu_lane"
echo "[$(date +%FT%T)] NIGHT HPO BATTERY COMPLETE"
