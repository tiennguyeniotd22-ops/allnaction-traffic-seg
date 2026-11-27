"""Apply image-specific decisions to Vietnam train-source polygons and CPU proposals."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from overnight_dataset import encode, sha
from overnight_drafts import COCO, draw, outline, to_rle


def geometry(points):
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "decisions", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    root = args.root
    folder = root / "rfvn"
    source = json.loads((folder / "index.json").read_text())
    teacher = json.loads(args.teacher.read_text()) if args.teacher else {}
    rules = json.loads(args.decisions.read_text())["windows"]
    records = []
    for sub in ("images", "labels", "views"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for row in source["records"]:
        key = row["key"]
        if key not in rules:
            continue
        rule = rules[key]
        path = folder / "images" / (key + ".jpg")
        body = path.read_bytes()
        label = (folder / "labels" / (key + ".txt")).read_bytes()
        assert sha(body) == row["image_sha256"] and sha(label) == row["label_sha256"]
        frame = Image.open(BytesIO(body)).convert("RGB")
        w, h = frame.size
        roi = rule["roi"]
        region = box(*roi)
        assert box(0, 0, w, h).covers(region) and region.area > 0
        cw, ch = roi[2]-roi[0], roi[3]-roi[1]
        shapes = {}
        for i, line in enumerate(label.decode().splitlines()):
            fields = line.split()
            cls = [0, 2, 1, 4][int(fields[0])]
            if str(i) in row["source_pickup_indices"]:
                assert cls == 2
                cls = 0
            shape = geometry(np.array(fields[1:], float).reshape(-1, 2) * [w, h])
            shapes["s"+str(i)] = (cls, shape)
        for i, pred in enumerate(teacher.get("images", {}).get(path.name, [])):
            if len(pred["polygon"]) >= 3:
                shapes["t"+str(i)] = (COCO[pred["label"]], geometry(pred["polygon"]))
        for name, (cls, points) in rule.get("trace", {}).items():
            shapes[name] = (cls, geometry(points))
        for name in rule.get("drop", []):
            del shapes[name]
        for name, cls in rule.get("reclass", {}).items():
            shapes[name] = (cls, shapes[name][1])
        for cls, names in rule.get("merge", []):
            shapes["+".join(names)] = (cls, unary_union([shapes.pop(n)[1] for n in names]))
        active = {n for n in shapes if not n.startswith("t") or "+" in n}
        active.update(rule.get("add_teacher", []))
        objects, lines = [], []
        for name in sorted(active):
            cls, polygon = shapes[name]
            cropped = polygon.intersection(region)
            if cropped.area < 0.5:
                continue
            pts = outline(cropped, w, h)
            if pts is None:
                if cropped.area * (640/max(cw,ch))**2 > 16:
                    raise ValueError((key, name, "lost visible surface"))
                continue
            pts = np.clip(pts-np.array(roi[:2]), [0, 0], [cw, ch])
            try:
                line = encode(cls, pts, cw, ch)
            except ValueError as error:
                original = to_rle(pts, cw, ch)
                rounded = np.round(pts / [cw, ch], 7) * [cw, ch]
                repaired = outline(geometry(rounded), cw, ch)
                if repaired is None:
                    raise ValueError((key, name)) from error
                line = encode(cls, repaired, cw, ch)
                pts = np.array(line.split()[1:], float).reshape(-1, 2) * [cw, ch]
                if float(mu.iou([to_rle(pts, cw, ch)], [original], [0])[0,0]) < .995:
                    raise ValueError((key, name, "loss of mask area during repair")) from error
            area = int(mu.area(to_rle(pts, cw, ch)))
            if area:
                objects.append(dict(id=name, cls=cls, area640=round(area*(640/max(cw,ch))**2,2), poly=pts.tolist()))
                lines.append(line)
        assert lines and len(lines)==len(set(lines)), key
        crop = frame.crop(roi)
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=95, subsampling=0)
        pixels = buf.getvalue()
        labels = ("\n".join(lines)+"\n").encode()
        (args.out/"images"/path.name).write_bytes(pixels)
        (args.out/"labels"/(key+".txt")).write_bytes(labels)
        draw(crop, objects).save(args.out/"views"/path.name, quality=94)
        records.append(dict(key=key, source=source["source"], url=row["image_url"], license=source["license"],
            revision=source["revision"], filename=row["filename"], source_split=row["source_split"],
            source_label_sha256=row["label_sha256"], parent=path.relative_to(root).as_posix(),
            parent_sha256=sha(body), published_sha256=row["published_manifest_sha256"], roi=roi, size=[cw,ch],
            teacher_sha256=teacher.get("teacher_sha256"), image_sha256=sha(pixels), label_sha256=sha(labels),
            objects=[{k:v for k,v in o.items() if k!="poly"} for o in objects]))
    assert len(records)==len(rules)
    (args.out/"manifest.json").write_text(json.dumps(records, indent=1))
    print(len(records), sum(len(r["objects"]) for r in records))


if __name__=="__main__":
    main()
