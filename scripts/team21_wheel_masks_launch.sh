#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
code=overnight/team21-roi-code
py=.venv/bin/python
while ! grep -qs PILOT_COMPLETE overnight/team21-wheel12.log; do
    tmux has-session -t =team21wheelfit || exit 1
    sleep 3
done
if "$py" "$code/team21_reached_target.py"; then exit 0; else [[ $? == 3 ]]; fi
parent=${1:-team21-mask-roi}
name=${2:-team21-wheel-contours}
version=${3:?verified dataset for the parent}
weights=overnight/weight-means/$name
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$py" -u "$code/team21_wheel_masks.py" "overnight/$parent" overnight/team21-wheel12 overnight/team21-wheel-reviewed "$weights" train_team21.ipynb
if [[ -s $weights/fit.json ]]; then
    TEAM21_CODE="$code" TEAM21_WHEEL_CONTOURS="$weights" bash "$code/team21_score_mean.sh" "$name" "$parent" "$version"
fi
