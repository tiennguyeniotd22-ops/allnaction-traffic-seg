#!/usr/bin/env bash
# Run a single controlled trial on the selected source ZIP and its matched baseline.
set -euo pipefail
cd ~/baseline-cvat300
recipe=${1:?experiment}
dataset=${2:?dataset path}
baseline=${3:?control run}
run=${4:?new output directory}
revision=${5:?code revision}
epochs=${6:-20}
patience=${7:-20}
code=overnight/ablation
test ! -e "$run"
test -s "$baseline/run_report.json"
expected_sha=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_sha256"])' "$baseline/resources.json")
echo "$expected_sha  $dataset" | sha256sum -c -
.venv/bin/python "$code/run_notebook_pilot.py" --dataset "$dataset" --run "$run" \
    --revision "$revision" --epochs "$epochs" --patience "$patience" --experiment "$recipe"
out=overnight/scores/${run##*/}
mkdir -p "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
nice -n 8 .venv/bin/python overnight/overnight_holdout.py --run "$run" \
    --holdout overnight/e2-v1/holdout.zip --manifest overnight/e2-v1/manifest.json \
    --notebook train_and_export.ipynb --out "$out/held"
nice -n 8 .venv/bin/python "$code/compare_pilot.py" --baseline "$baseline" --pilot "$run" \
    --manifest audit_phase12/validation-manifest.json --notebook train_and_export.ipynb \
    --out "$out/comparison.json" --experiment "$recipe"
