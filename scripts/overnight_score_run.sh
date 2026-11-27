#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
run=${1:?candidate run folder}
base=${2:?matched control folder}
schedule=${3:-same}
name=${run##*/}
out="overnight/scores/$name"
mkdir -p "$out"
while [[ ! -s "$run/run_report.json" || ! -s "$run/predictions_val.json" ]]; do sleep 15; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
if [[ ! -s "$out/held/score.json" ]]; then
  nice -n 8 .venv/bin/python overnight/overnight_holdout.py --run "$run" --holdout overnight/e2-v1/holdout.zip --manifest overnight/e2-v1/manifest.json --notebook train_and_export.ipynb --out "$out/held"
fi
extra=()
if [[ "$schedule" == "different" ]]; then extra+=(--allow-schedule-change); fi
if [[ ! -s "$out/comparison.json" ]]; then
  nice -n 8 .venv/bin/python round4/compare_pilot.py --baseline "$base" --pilot "$run" --manifest audit_phase12/validation-manifest.json --notebook train_and_export.ipynb --out "$out/comparison.json" "${extra[@]}"
fi
