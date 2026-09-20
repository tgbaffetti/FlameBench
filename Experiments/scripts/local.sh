#!/usr/bin/env bash
# One independent experiment per visible GPU. Run from the repository root.
set -euo pipefail
python_bin="${PYTHON:-python}"
pids=()
for spec in "0:gru" "1:lstm" "2:transformer"; do
    gpu="${spec%%:*}"
    model="${spec##*:}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m Experiments.run run --config "Experiments/Configs/pod_${model}.json" --seeds 0 1 2 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
