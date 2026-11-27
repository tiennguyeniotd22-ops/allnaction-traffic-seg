#!/usr/bin/env bash
set -euo pipefail
cd ~/baseline-cvat300
box=nuclio-nuclio-pth-facebookresearch-sam-vit-h
sudo -n docker cp round4/round4_source_masks.py "$box:/tmp/e3_vn_source_masks.py"
for kind in sign cam; do
  target="overnight/vn-$kind-sam.zip"
  if [[ -s "$target" ]]; then continue; fi
  input="vn-$kind-prompts.zip"
  sudo -n docker cp "overnight/$input" "$box:/tmp/$input"
  sudo -n docker exec "$box" nice -n 10 python /tmp/e3_vn_source_masks.py "/tmp/$input" "/tmp/e3-vn-$kind-sam.zip" > "overnight/vn-$kind-sam.log" 2>&1
  sudo -n docker cp "$box:/tmp/e3-vn-$kind-sam.zip" "$target.partial"
  sudo -n chown "$(id -u):$(id -g)" "$target.partial"
  .venv/bin/python -m zipfile -t "$target.partial"
  mv "$target.partial" "$target"
done
