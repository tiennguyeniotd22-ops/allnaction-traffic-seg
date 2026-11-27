#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
revision=${1:?source revision}
dataset=overnight/e3-tax-v2/dataset.zip
echo "d9fabdc563c4de14d4cb8205dcd10e6ffcc849186cfe69c8fa1a20b11efc0dad  $dataset" | sha256sum -c -
while [[ ! -s overnight/e3-vn20/run_report.json ]]; do
    tmux has-session -t '=e3pilot'
    sleep 15
done
test ! -e overnight/e3-tax20
.venv/bin/python round4/run_notebook_pilot.py --dataset "$dataset" \
    --run overnight/e3-tax20 --revision "$revision" --epochs 20 >overnight/e3-tax20.log 2>&1
bash overnight/score-run.sh overnight/e3-tax20 overnight/e3-vn20 >overnight/e3-tax20score.log 2>&1
