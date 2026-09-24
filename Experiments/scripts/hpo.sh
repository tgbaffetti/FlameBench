#!/usr/bin/env bash
# Benchmark protocol for each compressor-forecaster pair (see Experiments/run.py): HPO with the
# config seed, then the best config fitted and tested with each of $SEEDS (default 0-9). POD is
# fitted once and reused by every seed; POD + ARX is fully deterministic and runs one seed.
# "constant" is the constant baseline (identity compressor): nothing to tune, one seed.
#
# Usage, from the repository root inside a screen session:
#   Experiments/scripts/hpo.sh                  # every pair and the constant baseline
#   Experiments/scripts/hpo.sh pod_arx cae_gru  # only these
# Pairs are spread over the GPUs in $GPUS (default: every GPU nvidia-smi lists), $JOBS_PER_GPU
# pairs at a time per GPU (default: enough to start every pair at once; the small models leave
# the GPUs mostly idle). Each pair also runs "parallel" (config) trials or seeds at a time.
# One log per pair in logs/. A failed pair does not stop the others.
set -uo pipefail
python_bin="${PYTHON:-python}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
if (( $# )); then
    pairs=("$@")
else
    pairs=(constant pod_dmdc pod_opinf identity_fno)  # DMDc and OpInf are POD-only; FNO works on the full field.
    for compressor in pod cae vit_ae; do
        for forecaster in arx gru lstm cnn transformer; do
            pairs+=("${compressor}_${forecaster}")
        done
    done
fi
read -r -a gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
(( ${#gpus[@]} )) || { echo "No GPU found; set GPUS" >&2; exit 1; }
per_gpu=${JOBS_PER_GPU:-$(( (${#pairs[@]} + ${#gpus[@]} - 1) / ${#gpus[@]} ))}
slots=()  # one entry per concurrent worker, interleaved (0 1 2 0 1 2 ...) to mix heavy and light pairs
for (( k = 0; k < per_gpu; k++ )); do slots+=("${gpus[@]}"); done
mkdir -p logs

run_pair() {
    if [[ $1 == constant ]]; then
        "$python_bin" -m Experiments.run run --config Experiments/Configs/identity_constant.json
    else
        "$python_bin" -m Experiments.run hpo --config "Experiments/Configs/hpo_$1.json" --seeds "${seeds[@]}"
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
for i in "${!slots[@]}"; do
    share=()
    for (( j = i; j < ${#pairs[@]}; j += ${#slots[@]} )); do
        share+=("${pairs[$j]}")
    done
    (( ${#share[@]} )) && { worker "${slots[$i]}" "${share[@]}" & pids+=("$!"); }
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
