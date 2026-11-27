"""Propose same-class COCO geometry repairs, then build a pilot ZIP from accepted rows.

Inputs: frozen competition ZIP, COCO val2017 JSON (gzipped), versioned split manifest.
Review the generated crops. Pass a text file of accepted original image:row:annotation
keys to --accept. Identical labels on synthetic train siblings inherit each decision.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, unary_union

SOURCE_CLASS = {"car": 0, "bus": 1, "truck": 2, "stop sign": 9, "traffic light": 10}
COCO_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_outline(parts):
    """Connect nearby visible components with subpixel seams for single-ring YOLO.

    Every proposal is compared to the complete original RLE after serialization.
    Keep distant components and topologically ambiguous source masks for human QC.
    """
    shapes = [Polygon(np.array(part).reshape(-1, 2)) for part in parts]
    if not all(s.is_valid and s.area > 0 for s in shapes):
        return None
    shape = unary_union(shapes)
    while shape.geom_type == "MultiPolygon":
        components = list(shape.geoms)
        a, b = min(((a, b) for i, a in enumerate(components) for b in components[i+1:]),
                   key=lambda pair: pair[0].distance(pair[1]))
        if a.distance(b) > 32:
            return None
        endpoints = nearest_points(a, b)
        bridge = LineString(endpoints).buffer(.05)
        merged = shape.union(bridge)
        if merged.geom_type == "MultiPolygon" and len(merged.geoms) >= len(components):
            return None
        shape = merged
    if shape.geom_type != "Polygon" or not shape.is_valid:
        return None
    return np.asarray(shape.exterior.coords[:-1])


def crop_pair(image, old, new):
    width, height = image.size
    points = np.vstack((old, new)) * [width, height]
    lo = np.maximum(np.floor(points.min(0)) - 18, 0).astype(int)
    hi = np.minimum(np.ceil(points.max(0)) + 19, [width, height]).astype(int)
    panels = []
    for poly, color in [(old, "yellow"), (new, "lime")]:
        im = image.copy()
        xy = [tuple(p) for p in poly * [width, height]]
        ImageDraw.Draw(im).line(xy + [xy[0]], fill=color, width=2)
        im = im.crop((*lo, *hi))
        im.thumbnail((208, 140))
        panels.append(im)
    panel = Image.new("RGB", (422, 164), "#242424")
    for i, im in enumerate(panels):
        panel.paste(im, (212*i, 20))
    return panel


def propose(z, files, coco, out):
    cats = {c["id"]: SOURCE_CLASS[c["name"]] for c in coco["categories"] if c["name"] in SOURCE_CLASS}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if not ann["iscrowd"] and ann["category_id"] in cats:
            annotations[ann["image_id"]].append(ann)
    metadata = {r["id"]: r for r in coco["images"]}
    result = {}
    for item in files:
        name = item["file"]
        match = re.fullmatch(r"zz_drive_(\d+)\.jpg", name)
        if item["split"] != "train" or not match:
            continue
        idx = int(match[1])
        w, h = metadata[idx]["width"], metadata[idx]["height"]
        image_data = z.read("images/train/" + name)
        with Image.open(BytesIO(image_data)) as image:
            if image.size != (w, h):
                raise ValueError(name)
            old_lines = z.read(f"labels/train/{Path(name).stem}.txt").decode().splitlines()
            by_class = defaultdict(list)
            for i, line in enumerate(old_lines):
                c, *xy = line.split()
                poly = np.array(xy, dtype=float).reshape(-1, 2)
                mask = mu.frPyObjects([(poly * [w, h]).ravel().tolist()], h, w)[0]
                by_class[int(c)].append((i, poly, mask))
            for cls, old in by_class.items():
                anns = [a for a in annotations[idx] if cats[a["category_id"]] == cls]
                if not anns:
                    continue
                srcs = [mu.merge(mu.frPyObjects(a["segmentation"], h, w)) for a in anns]
                ious = mu.iou([o[2] for o in old], srcs, [0]*len(srcs))
                for j, (row, prev, _) in enumerate(old):
                    k = ious[j].argmax()
                    before = float(ious[j, k])
                    if not .45 <= before < .90 or j != ious[:, k].argmax():
                        continue
                    if np.any(np.delete(ious[j], k) > .20) or np.any(np.delete(ious[:, k], j) > .20):
                        continue
                    if mu.area(srcs[k]) * (640 / max(w, h))**2 < 64:
                        continue
                    ann = anns[k]
                    parts = ann["segmentation"]
                    outline = source_outline(parts)
                    if outline is None:
                        continue
                    poly = outline / [w,h]
                    if not np.isfinite(poly).all() or poly.min() < 0 or poly.max() > 1 or not Polygon(poly).is_valid:
                        continue
                    line = str(cls) + " " + " ".join(f"{v:.8f}" for v in poly.ravel())
                    new = np.array(line.split()[1:], dtype=float).reshape(-1, 2)
                    after_mask = mu.frPyObjects([(new*[w,h]).ravel().tolist()], h, w)[0]
                    after = float(mu.iou([after_mask], [srcs[k]], [0])[0, 0])
                    if after < .995 or not Polygon(new).is_valid:
                        continue
                    key = f"{name}:{row+1}:{ann['id']}"
                    panel = crop_pair(image.convert("RGB"), prev, new)
                    label = f"{len(result)+1}   class {cls}    IoU {before:.3f} -> {after:.3f}    {idx}/{ann['id']}"
                    ImageDraw.Draw(panel).text((3, 2), label, fill="white")
                    panel.save(out / f"candidate-{len(result)+1:03}.jpg", quality=94)
                    result[key] = dict(number=len(result)+1, image=name, row=row+1, annotation=ann["id"], cls=cls,
                                       source_image=idx, license_id=metadata[idx]["license"], source_url=metadata[idx]["coco_url"],
                                       source_parts=len(parts), before_iou=before, after_iou=after,
                                       old=old_lines[row], new=line)
    return result


def build(z, files, proposals, accepted, out):
    unknown = accepted - proposals.keys()
    if unknown or not accepted:
        raise ValueError(f"Invalid review selection: {unknown}, {len(accepted)} accepted")
    changes = defaultdict(list)
    by_name = {f["file"]:f for f in files}
    for key in sorted(accepted):
        prop = proposals[key]
        original = prop["image"]
        for name in [original, original.replace("zz_drive_", "zz_drive_syn_night_"),
                     original.replace("zz_drive_", "zz_drive_syn_rain_")]:
            if name not in by_name:
                continue
            meta = by_name[name]
            if meta["split"] != "train":
                raise ValueError("Sibling outside train: " + name)
            if meta["labels_sha256"] != by_name[original]["labels_sha256"]:
                continue
            changes[f"labels/train/{Path(name).stem}.txt"].append((key,prop))
    path = out / "pilot.zip"
    evidence = []
    with ZipFile(path.with_suffix(".tmp"), "x", compression=ZIP_DEFLATED, compresslevel=1) as dst:
        for name in z.namelist():
            data = z.read(name)
            if name in changes:
                lines = data.decode().splitlines(keepends=True)
                old_sha = digest(data)
                for key, obj in changes[name]:
                    i = obj["row"]-1
                    prefix = lines[i].rstrip("\r\n")
                    if prefix != obj["old"]:
                        raise ValueError(f"Source changed: {name}:{i+1}")
                    lines[i] = obj["new"] + lines[i][len(prefix):]
                data = "".join(lines).encode()
                for key, obj in changes[name]:
                    evidence.append(dict(key=key, file=name, row=obj["row"], cls=obj["cls"],
                                         source_annotation=obj["annotation"], before_iou=obj["before_iou"],
                                         after_iou=obj["after_iou"], old_sha256=old_sha, new_sha256=digest(data)))
            info = ZipInfo(name, (2026, 11, 27, 0, 0, 0))
            dst.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=1)
    temporary = path.with_suffix(".tmp")
    with ZipFile(temporary) as other:
        assert z.namelist() == other.namelist()
        changed = {n for n in z.namelist() if z.read(n) != other.read(n)}
        assert changed == changes.keys()
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    (out/"edits.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    print("accepted",len(accepted),"changes",len(evidence),"files",len(changed),"zip_sha256",digest(path.read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset","coco","split","out"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--accept", type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    files = json.loads(a.split.read_text(encoding="utf-8"))
    with ZipFile(a.dataset) as z:
        expected = {"data.yaml"}
        for row in files:
            for folder, ext, field in [("images", ".jpg", "image_sha256"), ("labels", ".txt", "labels_sha256")]:
                name = f"{folder}/{row['split']}/{Path(row['file']).stem}{ext}"
                expected.add(name)
                if digest(z.read(name)) != row[field]:
                    raise ValueError(name)
        if len(set(z.namelist())) != len(z.namelist()) or expected != set(z.namelist()):
            raise ValueError("Archive membership differs from the frozen split")
        data = gzip.decompress(a.coco.read_bytes())
        if digest(data) != COCO_SHA256:
            raise ValueError("COCO source fingerprint mismatch")
        props = propose(z, files, json.loads(data), a.out)
        (a.out/"proposals.json").write_text(json.dumps(props, indent=1), encoding="utf-8")
        print("candidate rows",len(props),"classes",Counter(x["cls"] for x in props.values()))
        if a.accept:
            accepted = set(a.accept.read_text(encoding="utf-8").splitlines())
            build(z, files, props, accepted, a.out)


if __name__ == "__main__":
    main()
