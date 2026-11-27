#!/usr/bin/env bash
# Each comparison uses the same taxonomy dataset, seed, and 20-epoch control.
set -euo pipefail
cd ~/baseline-cvat300
revision=${1:?training code revision}
while [[ ! -s overnight/e4-repeat20/run_report.json ]]; do
    tmux has-session -t '=e4repeat'
    sleep 15
done
bash overnight/ablation/overnight_trial.sh no-copy-paste overnight/e3-tax-v2/dataset.zip \
    overnight/e3-tax20 overnight/e4-nocopy20 "$revision" 20 20
