#!/usr/bin/env bash
# Export a single averaged model with the competition notebook and score the exact archive.
set -euo pipefail
cd ~/baseline-cvat300
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code=${TEAM21_CODE:-overnight/team21-transfer-code}
py=.venv/bin/python
run=$1
reference=$2
version=$3
shift 3
sources=()
for other in "$@"; do sources+=("overnight/$other"); done
dest=overnight/$run
base=overnight/$reference
scores=overnight/scores/$run
manifest=audit_phase12/validation-manifest.json
notebook=train_team21.ipynb
options=()
if [[ ${TEAM21_REFRESH_BN:-0} == 1 ]]; then options+=(--refresh-bn); fi
if [[ -n ${TEAM21_BN_MOMENTUM:-} ]]; then options+=(--bn-momentum "$TEAM21_BN_MOMENTUM"); fi
if [[ ${TEAM21_BN_NOTEBOOK_INPUT:-0} == 1 ]]; then options+=(--bn-use-notebook-input); fi
if [[ ${TEAM21_MASK_QUALITY:-0} == 1 ]]; then
    [[ ${#sources[@]} == 0 && -z ${TEAM21_PRETRAINED_PRIOR:-} && ${TEAM21_ONE2ONE_WEIGHTS:-0} == 0 ]]
    options+=(--mask-quality)
elif [[ ${TEAM21_ONE2ONE_WEIGHTS:-0} == 1 ]]; then
    [[ ${#sources[@]} == 0 && -z ${TEAM21_PRETRAINED_PRIOR:-} ]]
    options+=(--one2one-weights)
elif [[ -n ${TEAM21_PRETRAINED_PRIOR:-} ]]; then
    [[ ${#sources[@]} == 0 ]]
    options+=(--pretrained-prior "$TEAM21_PRETRAINED_PRIOR")
else
    options+=(--mean-with "${sources[@]}")
fi
"$py" "$code/overnight_last_checkpoint.py" "$base" "$dest" "$notebook" "$manifest" "${options[@]}"
mkdir -p "$scores"
if [[ ! -s "$scores/held/score.json" ]]; then
    "$py" "$code/overnight_holdout.py" --run "$dest" --holdout overnight/e2-v1/holdout.zip \
        --manifest overnight/e2-v1/manifest.json --notebook "$notebook" --out "$scores/held"
fi
compare=("$py" "$code/compare_pilot.py" --baseline "$base" --pilot "$dest" --notebook "$notebook" \
         --manifest "$manifest" --out "$scores/comparison.json")
if [[ ! -s "$scores/comparison.json" ]]; then
    if "${compare[@]}"; then
        true
    else
        status=$?
        [[ $status == 65 ]] || exit "$status"
        "$py" "$code/overnight_decode_parity.py" "$dest" "$notebook" "$manifest" "overnight/decoded/$run"
        "${compare[@]}" --decoded-parity "overnight/decoded/$run/decoded-comparison.json"
    fi
fi
bash overnight/team21-replay-fit.sh "$run" "$version"
