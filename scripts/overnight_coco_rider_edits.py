"""Rebuild reviewed train polygons from COCO IDs; write input to overnight_apply_vn_review.py."""
import argparse
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from overnight_apply_vn_review import file_sha
from overnight_dataset import encode, read, sha
from overnight_drafts import outline, to_rle
from prepare_source_pilot import COCO_SHA256, source_outline


def source_mask(annotations, ids, width, height):
    parts = [p for i in ids for p in annotations[i]["segmentation"]]
    return mu.decode(mu.merge(mu.frPyObjects(parts, height, width))).astype(bool)


def reviewed_polygon(annotations, ids, exclude, cls, width, height):
    foreground = source_mask(annotations, ids, width, height)
    parts = [p for i in ids for p in annotations[i]["segmentation"]]
    geom = unary_union([make_valid(Polygon(np.reshape(p, (-1, 2)))) for p in parts])
    candidates = []
    if exclude:
        foreground &= ~source_mask(annotations, exclude, width, height)
        occluders = [p for i in exclude for p in annotations[i]["segmentation"]]
        geom = geom.difference(unary_union([make_valid(Polygon(np.reshape(p, (-1, 2)))) for p in occluders]))
    else:
        candidates.append(source_outline(parts))
    candidates.append(outline(geom, width, height))
    target = mu.encode(np.asfortranarray(foreground.astype(np.uint8)))
    best = None

    def evaluate(points):
        if points is None:
            return None
        try:
            line = encode(cls, points, width, height)
        except ValueError:
            return None
        xy = np.reshape(np.asarray(line.split()[1:], float), (-1, 2)) * [width, height]
        iou = float(mu.iou([to_rle(xy, width, height)], [target], [0])[0, 0])
        return iou, line, xy

    for points in candidates:
        candidate = evaluate(points)
        if candidate and (best is None or candidate[0] > best[0]):
            best = candidate
    if best is None or best[0] < .99:
        raster = unary_union([make_valid(shape(s)) for s, _ in shapes(foreground.astype("uint8"), mask=foreground)])
        candidate = evaluate(outline(raster, width, height))
        if candidate and (best is None or candidate[0] > best[0]):
            best = candidate
    if best is None or best[0] < .985:
        raise ValueError(f"Source fidelity below .985 for {ids}: {best[0] if best else None}")
    iou, _, xy = best
    return dict(op="append", cls=cls, pixels=xy.tolist(), source_ids=ids,
                excluded_ids=exclude, source_iou=iou, source_pixels=int(mu.area(target)))


def main():
    p = argparse.ArgumentParser()
    for k in ("dataset", "source", "plan", "out"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    parent = file_sha(args.dataset)
    plan = read(args.plan)
    original_base = "f125ebfd5846fe6c948dd6c1bb33ce70c63cfb4fd7a505e7453e709d8af70a5e"
    if parent != plan.get("base_sha256", original_base):
        raise ValueError("Source rows must refer to the exact reviewed dataset")
    with ZipFile(args.source) as z:
        body = z.read("annotations/instances_val2017.json")
    assert sha(body) == COCO_SHA256
    annotations = {a["id"]: a for a in json.loads(body)["annotations"]}
    records = []
    with ZipFile(args.dataset) as z:
        for scene, entry in plan.get("coco", plan).items():
            suffix = f"{int(scene):012d}.jpg"
            original = "images/train/zz_drive_" + suffix
            sibling_names = [n for n in z.namelist() if n.startswith("images/") and n.endswith(suffix)]
            assert original in sibling_names and all(n.startswith("images/train/zz_drive_") for n in sibling_names)
            label = "labels/train/" + Path(original).stem + ".txt"
            rows = z.read(label).decode().splitlines()
            size = Image.open(BytesIO(z.read(original))).size
            additions = []
            for cls, ids, *rest in entry["add"]:
                exclude = rest[0] if rest else []
                assert all(annotations[i]["image_id"] == int(scene) and not annotations[i]["iscrowd"] for i in ids + exclude)
                additions.append(reviewed_polygon(annotations, ids, exclude, cls, *size))
            for image in sorted(sibling_names):
                raw = z.read(image)
                assert Image.open(BytesIO(raw)).size == size
                label = z.read("labels/train/" + Path(image).stem + ".txt")
                local = label.decode().splitlines()
                assert Counter(local) == Counter(rows) and len(local) == len(set(local))
                drops = [dict(op="delete", row=local.index(rows[i]), from_cls=int(rows[i].split()[0]),
                              old_row_sha256=sha(rows[i].encode())) for i in entry["remove"]]
                records.append(dict(image=image, size=size, image_sha256=sha(raw), label_sha256=sha(label),
                                    source="COCO val2017", coco_image_id=int(scene), reason=entry["reason"],
                                    actions=drops + additions))
        for name, entry in plan.get("bdd", {}).items():
            assert name.startswith("bdd_") and Path(name).name == name
            image = f"images/train/{name}.jpg"
            siblings = [n for n in z.namelist() if n.startswith("images/") and n.endswith(name + ".jpg")]
            assert siblings == [image]
            raw = z.read(image)
            label = z.read(f"labels/train/{name}.txt")
            line = label.decode().splitlines()[entry["row"]]
            assert int(line.split()[0]) == entry["from_cls"]
            records.append(dict(image=image, size=Image.open(BytesIO(raw)).size,
                                image_sha256=sha(raw), label_sha256=sha(label), source="BDD100K",
                                reason=entry["reason"], actions=[dict(op="relabel", row=entry["row"],
                                from_cls=entry["from_cls"], cls=entry["cls"], old_row_sha256=sha(line.encode()))]))
    report = dict(base_sha256=parent, source_json_sha256=COCO_SHA256, plan_sha256=file_sha(args.plan), records=records)
    args.out.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    fits = [a["source_iou"] for r in records for a in r["actions"] if "source_iou" in a]
    print(len(records), "train files;", len(fits), "new polygons; minimum source IoU", min(fits))


if __name__ == "__main__":
    main()
