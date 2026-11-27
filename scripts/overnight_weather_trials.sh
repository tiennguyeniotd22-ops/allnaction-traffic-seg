#!/usr/bin/env bash
# Finish the matched dataset pilot before testing independent training parameters.
set -euo pipefail
cd ~/baseline-cvat300
revision=${1:?training revision}
tax=overnight/e3-tax-v2/dataset.zip
base=overnight/e3-tax20
weather=overnight/e3-weather-v1/dataset.zip
echo "98c08b696635b776991aa26d8f0eafbbf8781c45a6bfb4c25dcbc9bbac06dbb8  $weather" | sha256sum -c -
while [[ ! -s overnight/e4-nocopy20/run_report.json ]]; do
    tmux has-session -t '=e4nocopy'
    sleep 15
done
run=overnight/e3-weather20
test ! -e "$run"
.venv/bin/python round4/run_notebook_pilot.py --dataset "$weather" --run "$run" \
    --revision "$revision" --epochs 20 >"$run.log" 2>&1
bash overnight/score-run.sh "$run" "$base" >"${run}score.log" 2>&1
for item in 'mask-ratio-4 e5-mask420' 'lower-lr e6-lowerlr20' 'linear-lr e6-linear20' 'sgd e6-sgd20'; do
    read -r recipe name <<<"$item"
    bash overnight/ablation/overnight_trial.sh "$recipe" "$tax" "$base" "overnight/$name" \
        "$revision" 20 20 >"overnight/$name.log" 2>&1
done
