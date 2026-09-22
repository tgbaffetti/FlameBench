#!/usr/bin/env bash
# Full protocol for each compressor-forecaster pair (see Experiments/run.py): HPO with the config
# seed, fit and test of the best config, then refit on all training data with seeds 0 1 2, each
# tested. "constant" is the constant baseline (identity compressor), which has nothing to tune.
#
# Usage, from the repository root inside a screen session:
#   Experiments/scripts/hpo.sh                  # every pair and the constant baseline
#   Experiments/scripts/hpo.sh pod_arx cae_gru  # only these
# Pairs are spread over the GPUs in $GPUS (default: every GPU nvidia-smi lists); each GPU runs
# its pairs one after another. One log per pair in logs/. A failed pair does not stop the others.
set -uo pipefail
python_bin="${PYTHON:-python}"
if (( $# )); then
    pairs=("$@")
else
    pairs=(constant)
    for compressor in pod cae vit_ae; do
        for forecaster in arx gru lstm cnn transformer; do
            pairs+=("${compressor}_${forecaster}")
        done
    done
fi
read -r -a gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
(( ${#gpus[@]} )) || { echo "No GPU found; set GPUS" >&2; exit 1; }
mkdir -p logs

run_pair() {
    if [[ $1 == constant ]]; then
        "$python_bin" -m Experiments.run run --config Experiments/Configs/identity_constant.json
    else
        "$python_bin" -m Experiments.run hpo --config "Experiments/Configs/hpo_$1.json" --seeds 0 1 2
    fi
}

worker() {  # worker <gpu> <pair>...: the pairs of one GPU, in order
    local gpu=$1 status=0 pair
    shift
    for pair in "$@"; do
        echo "$(date '+%F %T') GPU $gpu start $pair"
        if CUDA_VISIBLE_DEVICES="$gpu" run_pair "$pair" > "logs/hpo_${pair}.log" 2>&1; then
            echo "$(date '+%F %T') GPU $gpu done  $pair"
        else
            echo "$(date '+%F %T') GPU $gpu FAILED $pair (logs/hpo_${pair}.log)"
            status=1
        fi
    done
    return "$status"
}

# Background jobs of a script ignore Ctrl-C, so stop every worker and its Python explicitly.
trap 'trap - INT TERM; kill 0' INT TERM
pids=()
for i in "${!gpus[@]}"; do
    share=()
    for (( j = i; j < ${#pairs[@]}; j += ${#gpus[@]} )); do
        share+=("${pairs[$j]}")
    done
    (( ${#share[@]} )) && { worker "${gpus[$i]}" "${share[@]}" & pids+=("$!"); }
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
