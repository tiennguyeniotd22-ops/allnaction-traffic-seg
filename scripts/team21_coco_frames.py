"""Select new licensed train2017 frames for a full-frame review of weak traffic classes."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from overnight_dataset import sha
from overnight_drafts import COLORS
from overnight_extend import check_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    root = a.root
    pool = root / "team21-curation-pool"
    out = root / "team21-coco-frames-review"
    for name in ("images", "labels", "views", "sheets"):
        (out / name).mkdir(exist_ok=True, parents=True)
    prior = set()
    prior_ids = set()
    for archive in ("team21-roadframes-v1/dataset.zip", "e2-v1/holdout.zip", "probe-frozen.zip"):
        with ZipFile(root / archive) as z:
            for name in z.namelist():
                if Path(name).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                prior.add(sha(z.read(name)))
                prior_ids.update(re.findall(r"(?:zz_drive_|coco5_|cc)(\d{12})", name))
    for version in ("e2-v1", "e3-native-v1", "e3-v1", "team21-road-v1", "team21-context-v1"):
        rows = json.loads((root / version / "manifest.json").read_bytes())["records"]
        for row in rows:
            meta = row.get("source_meta", {})
            prior.add(row.get("parent_sha256") or meta.get("sha256"))
            if "coco_id" in meta:
                prior_ids.add(f"{int(meta['coco_id']):012}")
            prior_ids.update(re.findall(r"(?:coco5_|cc)(\d{12})", row.get("source_image", "")))
    for row in json.loads((root/"team21-paste-v1/manifest.json").read_bytes())["records"]:
        prior.add(row["donor_source_sha256"])
    wanted = {"truck": (2, 75), "motorcycle-unoccupied": (4, 45), "motorcycle-with-people": (5, 45),
              "bicycle-unoccupied": (6, 45), "bicycle-with-people": (7, 60), "traffic light": (10, 35)}
    counts = Counter()
    accepted = []
    rows = json.loads((pool / "manifest.json").read_bytes())
    for row in sorted(rows, key=lambda r: sha(("21/frame/" + r["key"]).encode())):
        bucket = row["bucket"]
        if bucket not in wanted or row["parent_sha256"] in prior:
            continue
        cls, limit = wanted[bucket]
        if counts[bucket] >= limit or row["source"] != "COCO train2017" or row["license"]["id"] not in (4,5):
            continue
        w, h = row["size"]
        key = row["key"]
        if re.fullmatch(r"cc\d{12}", key) is None or key[2:] in prior_ids:
            continue
        source = (pool / "labels" / (key + ".txt")).read_bytes()
        assert sha(source) == row["label_sha256"]
        kept, lines, polys = [], [], []
        for item, line in zip(row["objects"], source.decode().splitlines(), strict=True):
            poly = np.array(line.split()[1:], dtype=float).reshape(-1,2) * [w,h]
            if np.ptp(poly, axis=0).min() >= .0125 * max(w,h):
                kept.append(item)
                lines.append(line)
                polys.append(poly)
        if not 1 <= len(kept) <= 12 or min(o["iou"] for o in kept) < .95:
            continue
        if not any(obj["cls"] == cls and obj["area640"] >= 180 for obj in kept):
            continue
        body = ("\n".join(lines) + "\n").encode()
        try:
            check_labels(body, kept)
        except ValueError:
            continue
        data = (pool / "images" / (key + ".jpg")).read_bytes()
        assert sha(data) == row["image_sha256"] == row["parent_sha256"]
        (out / "images" / (key + ".jpg")).write_bytes(data)
        (out / "labels" / (key + ".txt")).write_bytes(body)
        frame = Image.open(BytesIO(data)).convert("RGB")
        draw = ImageDraw.Draw(frame)
        for item, poly in zip(kept, polys, strict=True):
            ring = list(map(tuple, poly))
            draw.line(ring + ring[:1], fill=COLORS[item["cls"]], width=2)
        frame.save(out / "views" / (key + ".jpg"), quality=94)
        accepted.append(dict(row, objects=kept, label_sha256=sha(body), source_label_sha256=sha(source)))
        counts[bucket] += 1
    accepted.sort(key=lambda r:(r["bucket"], r["key"]))
    (out / "manifest.json").write_bytes(json.dumps(accepted, indent=1).encode())
    for start in range(0,len(accepted),12):
        sheet = Image.new("RGB",(1520,720),"white")
        draw = ImageDraw.Draw(sheet)
        for j,row in enumerate(accepted[start:start+12]):
            x,y = j%4*380,j//4*240
            with Image.open(out / "views" / (row["key"] + ".jpg")) as frame:
                frame.thumbnail((377,219))
                sheet.paste(frame,(x+(380-frame.width)//2,y))
            draw.text((x+2,y+222),f"{start+j:03d}   "+" ".join(str(item["cls"]) for item in row["objects"]),fill="black")
        sheet.save(out / "sheets" / f"{start:03d}.jpg",quality=93)
    print(len(accepted), counts, flush=True)


if __name__ == "__main__":
    main()
