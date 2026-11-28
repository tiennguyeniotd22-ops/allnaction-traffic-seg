"""Prepare true-image context windows with COCO instance masks for visual acceptance."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu
from shapely.geometry import Polygon, box

from overnight_dataset import encode, sha
from overnight_drafts import COLORS, outline, to_rle


TARGETS = {"car": (0,), "bus": (1,), "truck": (2,), "bicycle-unoccupied": (6,),
           "motorcycle-unoccupied": (4,), "bicycle-with-people": (7,), "motorcycle-with-people": (5,),
           "traffic light": (10,), "stop sign": (9,)}


def prepare(folder, row, out):
    key = row["key"]
    w, h = row["size"]
    text = (folder / "labels" / (key + ".txt")).read_bytes()
    assert sha(text) == row["label_sha256"]
    shapes = []
    for line, item in zip(text.decode().splitlines(), row["objects"], strict=True):
        words = line.split()
        xy = np.array(words[1:], float).reshape(-1, 2) * [w, h]
        if not Polygon(xy).is_valid or item["iou"] < .95:
            return None
        shapes.append((item, Polygon(xy), xy))
    possible = [(geom.area, i) for i, (item, geom, _) in enumerate(shapes)
                if item["cls"] in TARGETS[row["bucket"]] and min(geom.bounds[2]-geom.bounds[0], geom.bounds[3]-geom.bounds[1]) >= .045 * max(w, h)]
    if not possible:
        return None
    _, i = max(possible)
    l, t, r, b = shapes[i][1].bounds
    cw = min(w, max(140, round((r-l)*1.70), round((b-t)*1.65)))
    ch = min(h, max(125, round(cw*.76), round((b-t)*1.50)))
    x = max(0, min(w-cw, round((l+r-cw)/2)))
    y = max(0, min(h-ch, round((t+b-ch)/2)))
    window = box(x, y, x+cw, y+ch)
    if not window.covers(shapes[i][1]):
        return None
    lines, records, polys = [], [], []
    for item, shape, xy in shapes:
        clipped = shape.intersection(window)
        if clipped.is_empty or clipped.area < 1:
            continue
        left, top, right, bottom = clipped.bounds
        if min(right-left, bottom-top) < .0125*max(cw, ch):
            continue
        if clipped.geom_type != "Polygon" or clipped.interiors:
            return None
        points = np.asarray(clipped.exterior.coords[:-1]) - [x, y]
        line = encode(item["cls"], points, cw, ch)
        decoded = np.asarray(line.split()[1:], float).reshape(-1, 2) * [cw, ch]
        native = mu.decode(to_rle(xy, w, h))[y:y+ch, x:x+cw]
        native_rle = mu.encode(np.asfortranarray(native))
        agreement = float(mu.iou([to_rle(decoded, cw, ch)], [native_rle], [0])[0,0])
        if agreement < .98 or not Polygon(decoded).is_valid:
            return None
        lines.append(line)
        records.append(dict(item, crop_iou=round(agreement,6)))
        polys.append((item["cls"], decoded))
    if not 1 <= len(records) <= 7 or not any(r["cls"] in TARGETS[row["bucket"]] for r in records):
        return None
    original = (folder / "images" / (key + ".jpg")).read_bytes()
    assert sha(original) == row["parent_sha256"]
    with Image.open(BytesIO(original)) as im:
        patch = im.convert("RGB").crop((x, y, x+cw, y+ch))
    buf = BytesIO()
    patch.save(buf, "JPEG", quality=96, subsampling=0)
    data = buf.getvalue()
    labels = ("\n".join(lines)+"\n").encode()
    name = "roi_" + key
    (out / "images" / (name+".jpg")).write_bytes(data)
    (out / "labels" / (name+".txt")).write_bytes(labels)
    overlay = patch.copy()
    draw = ImageDraw.Draw(overlay)
    for cls, points in polys:
        ring = list(map(tuple, points))
        draw.line(ring + ring[:1], fill=COLORS[cls], width=max(2, round(max(cw,ch)/230)))
    overlay.thumbnail((440,330))
    overlay.save(out / "views" / (name+".jpg"), quality=94)
    return dict(row, key=name, image_sha256=sha(data), label_sha256=sha(labels), size=[cw,ch],
                roi=[x,y,x+cw,y+ch], parent_size=[w,h], objects=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--per-bucket", type=int, default=65)
    args = parser.parse_args()
    pool = args.root / "team21-curation-pool"
    rows = json.loads((pool/"manifest.json").read_bytes())
    prior = set()
    with ZipFile(args.root/"team21-wideqc-v1/dataset.zip") as z:
        prior.update(sha(z.read(n)) for n in z.namelist() if n.startswith("images/") and n.endswith(".jpg"))
    for src in ("e2-v1", "e3-tax-v2", "team21-road-v1"):
        for r in json.loads((args.root/src/"manifest.json").read_bytes()).get("records", []):
            prior.add(r.get("parent_sha256") or r.get("source_meta",{}).get("sha256"))
    for folder in ("images","labels","views"):
        (args.out/folder).mkdir(parents=True,exist_ok=True)
    counts, kept = Counter(), []
    for row in sorted(rows,key=lambda r: sha(("21/"+r["key"]).encode())):
        bucket = row["bucket"]
        if row["parent_sha256"] in prior or counts[bucket] >= args.per_bucket:
            continue
        candidate = prepare(pool,row,args.out)
        if candidate:
            kept.append(candidate)
            counts[bucket] += 1
    (args.out/"manifest.json").write_bytes(json.dumps(kept,indent=1).encode())
    print(len(kept),counts)


if __name__ == "__main__":
    main()
