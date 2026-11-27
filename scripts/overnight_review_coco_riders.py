"""Render the original COCO train riders with their current polygons and source human/bike IDs."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as masks

from overnight_drafts import COLORS


def draw(im, parts, text, color):
    pen = ImageDraw.Draw(im)
    points = []
    for part in parts:
        xy = list(map(tuple, np.asarray(part).reshape(-1, 2)))
        pen.line(xy + xy[:1], fill=color, width=2)
        points.extend(xy)
    x, y = np.asarray(points).min(axis=0)
    pen.text((x, y), text, fill=color, stroke_width=1, stroke_fill="black")


def main():
    p = argparse.ArgumentParser()
    for key in ("dataset", "coco", "out"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    with ZipFile(args.coco) as z:
        coco = json.loads(z.read("annotations/instances_val2017.json"))
    by_image = defaultdict(list)
    for obj in coco["annotations"]:
        if obj["category_id"] in (1, 2, 4) and not obj["iscrowd"] and isinstance(obj["segmentation"], list):
            by_image[obj["image_id"]].append(obj)
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = args.out / "scenes"
    scenes.mkdir(exist_ok=True)
    records = []
    with ZipFile(args.dataset) as z:
        for name in sorted(z.namelist()):
            match = re.fullmatch(r"images/train/zz_drive_(\d{12})\.jpg", name)
            if not match:
                continue
            key = int(match[1])
            src = by_image[key]
            raw = Image.open(BytesIO(z.read(name))).convert("RGB")
            w, h = raw.size
            original = raw.copy()
            bikes = []
            for obj in src:
                cat = obj["category_id"]
                color = {1: "#32e8fd", 2: "#a6ea38", 4: "#ff82a7"}[cat]
                draw(original, obj["segmentation"], str(obj["id"]), color)
                if cat != 1 and obj["area"] * (640 / max(w, h)) ** 2 >= 64:
                    bikes.append(obj)
            if not bikes:
                continue
            current = raw.copy()
            lines = z.read(f"labels/train/{Path(name).stem}.txt").decode().splitlines()
            target = []
            for i, line in enumerate(lines):
                cls, *v = line.split()
                cls = int(cls)
                if cls not in (4, 5, 6, 7, 8):
                    continue
                poly = np.asarray(v, dtype=float).reshape(-1, 2) * [w, h]
                draw(current, [poly], f"{i}/{cls}", COLORS[cls])
                target.append((i, cls, masks.frPyObjects([poly.ravel().tolist()], h, w)[0]))
            current.save(scenes / f"{key}-labels.jpg", quality=94)
            original.save(scenes / f"{key}-source.jpg", quality=94)
            raw.save(scenes / f"{key}-raw.jpg", quality=94)
            for obj in bikes:
                x, y, bw, bh = obj["bbox"]
                pad = max(18, .17 * max(bw, bh))
                window = (max(0, int(x - pad)), max(0, int(y - max(1.2 * bh, 40))),
                          min(w, int(x + bw + pad + 1)), min(h, int(y + bh + pad + 1)))
                source = masks.merge(masks.frPyObjects(obj["segmentation"], h, w))
                overlaps = masks.iou([a[2] for a in target], [source], [0]).flatten() if target else []
                best = int(np.argmax(overlaps)) if len(overlaps) else None
                index = len(records)
                records.append(dict(index=index, image=name, source_id=obj["id"], category=obj["category_id"],
                                    source_box=obj["bbox"], crop=window,
                                    best_row=target[best][0] if best is not None else None,
                                    best_cls=target[best][1] if best is not None else None,
                                    bike_only_iou=float(overlaps[best]) if best is not None else 0))
                if index % 9 == 0:
                    sheet = Image.new("RGB", (1140, 3 * 255), "#ddd")
                ox, oy = index % 3 * 380, index % 9 // 3 * 255
                for j, im in enumerate([raw, current]):
                    crop = im.crop(window)
                    ratio = min(188/crop.width, 218/crop.height)
                    crop = crop.resize((max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))))
                    sheet.paste(crop, (ox+j*190, oy))
                ImageDraw.Draw(sheet).text((ox+1, oy+221), f"{index}  image {key}  ann {obj['id']}  row {records[-1]['best_row']}", fill="black")
                if index % 9 == 8:
                    sheet.save(args.out / f"{index//9:03}.jpg", quality=93)
    if records and len(records) % 9:
        sheet.save(args.out / f"{(len(records)-1)//9:03}.jpg", quality=93)
    (args.out/"index.json").write_text(json.dumps(records,indent=1)+"\n")
    print(len(records), "bike crops from", len({r["image"] for r in records}), "original training images")


if __name__ == "__main__":
    main()
