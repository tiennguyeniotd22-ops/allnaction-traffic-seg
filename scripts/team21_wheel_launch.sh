#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
code=overnight/team21-roi-code
py=.venv/bin/python
fit=team21-wheel12
version=team21-wheel-focus-v1
# The preceding notebook also uses CUDA during validation and export.
until grep -qs PILOT_COMPLETE overnight/team21-balanced20.log; do sleep 3; done
if "$py" "$code/team21_reached_target.py"; then exit 0; else [[ $? == 3 ]]; fi
if [[ ! -s overnight/$version/manifest.json ]]; then
    "$py" "$code/team21_wheel_subset.py" overnight/team21-paste-v1/dataset.zip audit_phase12/validation-manifest.json "overnight/$version"
fi
[[ ! -e overnight/$fit ]]
"$py" -u "$code/run_notebook_pilot.py" --dataset "overnight/$version/dataset.zip" --run "overnight/$fit" \
    --revision "${TEAM21_REVISION:?}" --notebook train_team21.ipynb --warm-start overnight/team21-mask-roi \
    --epochs 12 --patience 12 --snapshot-every 4 --experiment warm-polish >"overnight/$fit.log" 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 8 "$py" "$code/overnight_finish_selection.py" --fit "$fit" --version "$version" --reference team21-paste20 --train-subset >"overnight/$fit-score.log" 2>&1
if [[ ! -s "overnight/probe-scores/$fit/score.json" ]]; then
    nice -n 8 bash overnight/team21-replay-fit.sh "$fit" "$version" > "overnight/$fit-replay.log" 2>&1
fi
