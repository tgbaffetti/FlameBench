#!/usr/bin/env bash
# Smoke battery: every core pair once, fixed hand-set configs, no HPO, single seed (42).
# One sequential lane per GPU plus a CPU lane, cheapest first; each run = fit + full test
# (free AR rollout + one-step protocol per the config's evaluation section).
#
# Launch DETACHED from the repository root (apptainer must stay in the foreground of the
# detached shell — `nohup apptainer &` bus-errors on squashfuse unmount):
#   mkdir -p Experiments/Results
#   setsid nohup bash Experiments/scripts/smoke.sh > Experiments/Results/smoke.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/../.."
repo="$PWD"
sif="$HOME/miniconda.sif"
python="$HOME/envs/dtd2c/bin/python"

run() { # run <gpu|cpu> <config-stem> [extra args...]
    local gpu="$1" cfg="$2"; shift 2
    local log="Experiments/Results/smoke_${cfg}.log"
    echo "[$(date +%FT%T)] START ${cfg} (gpu=${gpu})"
    if [ "$gpu" = "cpu" ]; then
        apptainer exec --bind /srv/mlg --env "PYTHONPATH=${repo}" --env CUDA_VISIBLE_DEVICES= \
            "$sif" "$python" -m Experiments.run run --config "Experiments/Configs/${cfg}.json" \
            --device cpu "$@" > "$log" 2>&1
    else
        apptainer exec --nv --bind /srv/mlg --env "PYTHONPATH=${repo}" --env "CUDA_VISIBLE_DEVICES=${gpu}" \
            "$sif" "$python" -m Experiments.run run --config "Experiments/Configs/${cfg}.json" \
            "$@" > "$log" 2>&1
    fi
    echo "[$(date +%FT%T)] DONE  ${cfg} exit=$? (log: ${log})"
}

lane() { local gpu="$1"; shift; for cfg in "$@"; do run "$gpu" "$cfg"; done; }

lane cpu identity_constant pod_arx pod_narx &
cpu_lane=$!
lane 0 pod_gru pod_lstm pod_transformer cae_transformer &
lane0=$!
lane 1 cae_arx cae_narx cae_gru cae_lstm &
lane1=$!
lane 2 identity_deeponet identity_transolver &
lane2=$!

wait "$cpu_lane" "$lane0" "$lane1" "$lane2"
echo "[$(date +%FT%T)] SMOKE BATTERY COMPLETE"
