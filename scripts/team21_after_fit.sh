#!/usr/bin/env bash
# Launch a matched experiment after the preceding notebook has finished using the GPU.
set -euo pipefail
cd ~/baseline-cvat300
previous=$1
session=$2
run=$3
version=$4
reference=$5
recipe=$6
revision=$7
code=${8:-overnight/team21-code}
while ! grep -qs PILOT_COMPLETE "overnight/$previous.log"; do
    tmux has-session -t "=$session" || { echo "Previous fit exited before export"; exit 1; }
    sleep 5
done
if .venv/bin/python "$code/team21_reached_target.py"; then
    echo "Verified target reached. Skipping $run."
    exit 0
else
    status=$?
    [[ $status == 3 ]] || exit "$status"
fi
test ! -e "overnight/$run"
options=()
epochs=${TEAM21_EPOCHS:-20}
if [[ -n ${TEAM21_WARM_START:-} ]]; then options+=(--warm-start "overnight/$TEAM21_WARM_START"); fi
if [[ -n ${TEAM21_SNAPSHOT:-} ]]; then options+=(--snapshot-every "$TEAM21_SNAPSHOT"); fi
.venv/bin/python -u "$code/run_notebook_pilot.py" --dataset "overnight/$version/dataset.zip" \
    --run "overnight/$run" --revision "$revision" --notebook train_team21.ipynb \
    --epochs "$epochs" --patience "$epochs" --experiment "$recipe" "${options[@]}" > "overnight/$run.log" 2>&1
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Wait for the reference comparison if scoring runs concurrently with the next fit.
while [[ ! -s "overnight/scores/$reference/comparison.json" ]]; do
    tmux has-session -t "=$session" || { echo "Reference comparison missing"; exit 1; }
    sleep 5
done
nice -n 8 .venv/bin/python -u "$code/overnight_finish_selection.py" --fit "$run" --reference "$reference" \
    --version "$version" > "overnight/$run-score.log" 2>&1
if [[ ! -s "overnight/replay-cpu-$run/replay.json" ]]; then
    nice -n 8 bash overnight/team21-replay-fit.sh "$run" "$version" > "overnight/$run-replay.log" 2>&1
fi
