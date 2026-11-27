"""Stage auditable COCO and panoptic examples for a full-window visual review."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as masks

from overnight_dataset import encode, read, sha
from overnight_drafts import to_rle


def coco(root, per_bucket):
    folder = root / "coco-extra"
    with ZipFile(root / "coco-more.zip") as z:
        truth = {r["coco_id"]: r for r in json.loads(z.read("manifest.json"))}
    buckets = Counter()
    for row in read(folder / "index.json"):
        meta = row["meta"]
        bucket = meta["selection"]
        if buckets[bucket] >= per_bucket:
            continue
        anns = {str(r["id"]): r for r in truth[meta["coco_id"]]["objects"]}
        objects = read(folder / "objects" / (row["key"] + ".json"))
        if not 1 <= len(objects) <= 14:
            continue
        h, w = row["h"], row["w"]
        consumed = []
        rows, lines = [], []
        for obj in objects:
            parts = obj.get("components", [obj])
            ids = [p["via"].split()[-1] for p in parts]
            consumed.extend(ids)
            native = masks.merge([masks.merge(masks.frPyObjects(anns[i]["segmentation"], h, w)) for i in ids])
            try:
                line = encode(obj["cls"], obj["poly"], w, h)
            except ValueError:
                break
            decoded = np.asarray(line.split()[1:], float).reshape(-1, 2) * [w, h]
            fit = float(masks.iou([to_rle(decoded, w, h)], [native], [0])[0, 0])
            if fit < (.95 if obj["area640"] >= 64 else .85):
                break
            lines.append(line)
            rows.append(dict(cls=obj["cls"], native_ids=ids, area640=obj["area640"], iou=round(fit, 5)))
        else:
            if len(consumed) != len(set(consumed)) or set(consumed) != set(anns):
                continue
            parent = folder / "images" / (row["key"] + ".jpg")
            body = parent.read_bytes()
            assert sha(body) == meta["sha256"]
            label = ("\n".join(lines) + "\n").encode()
            item = dict(key=f"cc{meta['coco_id']:012d}", source="COCO train2017", source_image=meta["file"],
                        url=meta["url"], original_url=meta["flickr_source"], license=meta["license"],
                        parent=parent.relative_to(root).as_posix(), parent_sha256=meta["sha256"],
                        image_sha256=sha(body), label_sha256=sha(label), size=[w, h], objects=rows, bucket=bucket)
            buckets[bucket] += 1
            yield item, body, label, folder / "views" / (row["key"] + ".jpg")


def native(root, kind, selection):
    folder = root / (kind + "-curated")
    index = read(root / kind / "index.json")
    for row in read(folder / "manifest.json"):
        key = row["key"]
        if "roi" not in row or selection is not None and int(key[2:]) not in selection:
            continue
        parent = root / kind / "images" / (key + ".png")
        body = (folder / "images" / (key + ".jpg")).read_bytes()
        label = (folder / "labels" / (key + ".txt")).read_bytes()
        assert sha(body) == row["image_sha256"] and sha(label) == row["label_sha256"]
        item = dict(row, source=index["source"], url=index.get("source_page") or index.get("homepage"),
                    license=index["license"], parent=parent.relative_to(root).as_posix(),
                    parent_sha256=row["source_sha256"], bucket=kind)
        yield item, body, label, folder / "views" / (key + ".jpg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--coco-per-bucket", type=int, default=0)
    p.add_argument("--kitti", type=Path)
    p.add_argument("--muses", action="store_true")
    a = p.parse_args()
    sources = []
    if a.coco_per_bucket:
        sources.append(coco(a.root, a.coco_per_bucket))
    if a.kitti:
        sources.append(native(a.root, "kitti", set(read(a.kitti)["keep"])))
    if a.muses:
        sources.append(native(a.root, "muses", None))
    for sub in ("images", "labels", "views"):
        (a.out / sub).mkdir(exist_ok=True, parents=True)
    records = []
    for source in sources:
        for row, body, label, view in source:
            key = row["key"]
            (a.out / "images" / (key + ".jpg")).write_bytes(body)
            (a.out / "labels" / (key + ".txt")).write_bytes(label)
            (a.out / "views" / (key + ".jpg")).write_bytes(view.read_bytes())
            records.append(row)
    (a.out / "manifest.json").write_text(json.dumps(records, indent=1))
    print(len(records), Counter(r["bucket"] for r in records))


if __name__ == "__main__":
    main()
