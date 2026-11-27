#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
revision=${1:?source revision}
cd overnight/e3-v1
echo "f0bb1888b238d40e8c4e0b3367e1486d3afadde3bff278cadea778c09963a667  dataset.zip" | sha256sum -c -
cd ../..
while tmux has-session -t '=e2queue' 2>/dev/null; do sleep 15; done
test -s overnight/e2-full100/run_report.json
test ! -e overnight/e3-vn20
.venv/bin/python round4/run_notebook_pilot.py --dataset overnight/e3-v1/dataset.zip \
    --run overnight/e3-vn20 --revision "$revision" --epochs 20 >overnight/e3-vn20.log 2>&1
bash overnight/score-run.sh overnight/e3-vn20 overnight/e2-pilot20 >overnight/e3-vn20score.log 2>&1
