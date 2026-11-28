#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
code=overnight/team21-rank-code
py=.venv/bin/python
parent=team21-mask-roi
run=team21-wheel-rank
weights=overnight/weight-means/$run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if "$py" "$code/team21_reached_target.py"; then exit 0; else [[ $? == 3 ]]; fi
"$py" -u "$code/team21_wheel_rank.py" "overnight/$parent" overnight/team21-wheel12 overnight/team21-wheel-reviewed "$weights" train_team21.ipynb
if [[ -s $weights/fit.json ]]; then
    TEAM21_CODE="$code" TEAM21_WHEEL_CONTOURS="$weights" bash "$code/team21_score_mean.sh" "$run" "$parent" e3-triage-v1
fi
