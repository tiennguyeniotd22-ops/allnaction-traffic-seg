#!/usr/bin/env bash
# Resume scientific checks directly from an already exported model.
set -euo pipefail
cd ~/baseline-cvat300
run=${1:?run}
parent=${2:?parent run}
version=${3:?dataset}
code=${TEAM21_CODE:-overnight/team21-rank-code}
py=.venv/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=overnight/scores/$run
mkdir -p "$out"
if [[ ! -s $out/held/score.json ]]; then
    "$py" "$code/overnight_holdout.py" --run "overnight/$run" --holdout overnight/e2-v1/holdout.zip --manifest overnight/e2-v1/manifest.json --notebook train_team21.ipynb --out "$out/held"
fi
if [[ ! -s $out/comparison.json ]]; then
    compare=("$py" "$code/compare_pilot.py" --baseline "overnight/$parent" --pilot "overnight/$run" --notebook train_team21.ipynb --manifest audit_phase12/validation-manifest.json --out "$out/comparison.json")
    if "${compare[@]}"; then :; else
        status=$?
        [[ $status == 65 ]] || exit "$status"
        "$py" "$code/overnight_decode_parity.py" "overnight/$run" train_team21.ipynb audit_phase12/validation-manifest.json "overnight/decoded/$run"
        "${compare[@]}" --decoded-parity "overnight/decoded/$run/decoded-comparison.json"
    fi
fi
bash overnight/team21-replay-fit.sh "$run" "$version"
