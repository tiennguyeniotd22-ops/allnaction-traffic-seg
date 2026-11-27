"""Build contextual train-only zoom candidates and compare each clipped mask to its parent."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu
from shapely.geometry import Polygon, box

from overnight_dataset import encode, sha
from overnight_drafts import COLORS, to_rle


def crop_candidate(z, image_name, target):
    key = Path(image_name).stem
    image_bytes = z.read(image_name)
    original = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = original.size
    label_name = f"labels/train/{key}.txt"
    labels = z.read(label_name)
    instances = []
    choices = []
    for index, line in enumerate(labels.decode().splitlines()):
        fields = line.split()
        cls = int(fields[0])
        xy = np.asarray(fields[1:], float).reshape(-1, 2) * [w, h]
        geom = Polygon(xy)
        if not geom.is_valid or geom.area <= 0:
            return
        instances.append((index, cls, xy, geom))
        area = geom.area * (640 / max(w, h))**2
        if cls == target and 40 <= area <= 6200:
            choices.append((abs(np.log(area / 480)), index, geom))
    if not choices:
        return
    _, focus, selected = min(choices, key=lambda item: item[:2])
    left, top, right, bottom = selected.bounds
    for scale in (0.54, 0.65):
        cw, ch = round(w * scale), round(h * scale)
        if cw < 200 or ch < 155 or right-left > .7*cw or bottom-top > .7*ch:
            continue
        x = max(0, min(w-cw, round((left+right-cw)/2)))
        y = max(0, min(h-ch, round((top+bottom-ch)/2)))
        rect = box(x, y, x+cw, y+ch)
        if not rect.covers(selected):
            continue
        output, objects, polys = [], [], []
        for index, cls, xy, geom in instances:
            if not geom.intersects(rect):
                continue
            clip = geom.intersection(rect)
            target_mask = mu.decode(to_rle(xy, w, h))[y:y+ch, x:x+cw]
            if not target_mask.any():
                continue
            if clip.geom_type != "Polygon" or clip.interiors:
                break
            coords = np.asarray(clip.exterior.coords[:-1]) - [x, y]
            try:
                line = encode(cls, coords, cw, ch)
            except ValueError:
                break
            decoded = np.asarray(line.split()[1:], float).reshape(-1, 2) * [cw, ch]
            ref = mu.encode(np.asfortranarray(target_mask))
            iou = mu.iou([to_rle(decoded, cw, ch)], [ref], [0])[0, 0]
            if iou < .985:
                break
            output.append(line)
            objects.append(dict(cls=cls, parent_row=index, raster_iou=float(iou)))
            polys.append((cls, coords))
        else:
            if not any(o["parent_row"] == focus for o in objects) or len(objects) > 26:
                continue
            patch = original.crop((x, y, x+cw, y+ch))
            stream = BytesIO()
            patch.save(stream, format="JPEG", quality=96, subsampling=0)
            data = stream.getvalue()
            label = ("\n".join(output)+"\n").encode()
            view = patch.copy()
            draw = ImageDraw.Draw(view)
            for cls, pts in polys:
                ring = [tuple(v) for v in pts]
                draw.line(ring+[ring[0]], fill=COLORS[cls], width=2)
            record = dict(key="focus_"+key, parent_image=image_name, parent_label=label_name,
                          parent_sha256=sha(image_bytes), parent_label_sha256=sha(labels), focus=focus,
                          target=target, roi=[x,y,x+cw,y+ch], size=[cw,ch], image_sha256=sha(data),
                          label_sha256=sha(label), objects=objects)
            return record, data, label, view


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--per-class", type=int, default=20)
    a = p.parse_args()
    for sub in ("images", "labels", "views"):
        (a.out/sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(26)
    used, records = set(), []
    with ZipFile(a.dataset) as z:
        image_names = sorted(n for n in z.namelist() if n.startswith(("images/train/bdd_", "images/train/vn_", "images/train/zz_drive_"))
                             and n.endswith(".jpg") and "_syn_" not in n)
        for target in (6, 4, 7, 9, 10, 5):
            rng.shuffle(image_names)
            count = 0
            for name in image_names:
                if name in used:
                    continue
                label = z.read("labels/train/"+Path(name).stem+".txt").decode().splitlines()
                if not any(row.split()[0] == str(target) for row in label):
                    continue
                output = crop_candidate(z, name, target)
                if output is None:
                    continue
                record, data, label_bytes, preview = output
                key = record["key"]
                (a.out/"images"/(key+".jpg")).write_bytes(data)
                (a.out/"labels"/(key+".txt")).write_bytes(label_bytes)
                preview.save(a.out/"views"/(key+".jpg"), quality=92)
                records.append(record)
                used.add(name)
                count += 1
                if count == a.per_class:
                    break
    (a.out/"manifest.json").write_text(json.dumps(records, indent=1)+"\n")
    print(len(records), Counter(row["target"] for row in records))


if __name__ == "__main__":
    main()
