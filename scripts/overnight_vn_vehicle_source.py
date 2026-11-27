"""Stage the train portion of a licensed Vietnam vehicle polygon selection."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from overnight_dataset import sha


REV = "de7f424f4ea205f5de6bb5243e06dd2d21a585ff"
ROOT = f"https://raw.githubusercontent.com/levienbang/yolo11/{REV}/datasets/"
SUB = "vietnam-seg-review-100/"
SOURCE = "https://universe.roboflow.com/phan-ky-dat/vietnam-traffic-vehicle-detectio/dataset/1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    for name in ("images", "labels", "views"):
        (args.out / name).mkdir(exist_ok=True, parents=True)

    def fetch(url, path):
        if not path.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=30) as response:
                        data = response.read()
                    temp = path.with_suffix(path.suffix + ".partial")
                    temp.write_bytes(data)
                    temp.replace(path)
                    break
                except OSError:
                    if attempt == 2:
                        raise
        return path.read_bytes()

    manifest = fetch(ROOT + SUB + "manifest.csv", args.out / "source.csv")
    relabel = fetch(ROOT + "pickup_relabel.csv", args.out / "pickup.csv")
    license_text = fetch(ROOT + SUB + "README.dataset.txt", args.out / "license.txt")
    assert b"CC BY 4.0" in license_text
    tree = json.loads(fetch(f"https://api.github.com/repos/levienbang/yolo11/git/trees/{REV}?recursive=1",
                            args.out / "git-tree.json"))
    assert not tree["truncated"]
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    original = list(csv.DictReader(StringIO(manifest.decode("utf-8-sig"))))
    rows = [row for row in original if row["source_split"] == "train"]
    pickups = [r for r in csv.DictReader(StringIO(relabel.decode("utf-8-sig"))) if r["dataset"] == SUB[:-1]]

    def stage(item):
        index, row = item
        key = f"rfvn{index:03d}"
        im = fetch(ROOT + SUB + row["image_path"], args.out / "images" / (key + ".jpg"))
        label = fetch(ROOT + SUB + row["label_path"], args.out / "labels" / (key + ".txt"))
        for content, field in ((im, "image_path"), (label, "label_path")):
            git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            assert git_sha == blobs["datasets/" + SUB + row[field]]
        image = Image.open(BytesIO(im)).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)
        objects = []
        for i, line in enumerate(label.decode().splitlines()):
            fields = line.split()
            source_cls = int(fields[0])
            xy = np.array(fields[1:], float).reshape(-1, 2)
            assert np.isfinite(xy).all() and xy.min() >= 0 and xy.max() <= 1 and len(xy) >= 3
            points = [tuple(p) for p in xy * [w, h]]
            p = Polygon(points)
            objects.append(dict(line=i, source_class=source_cls, n=len(xy), valid=p.is_valid,
                                rectangle_fit=round(p.area / max(p.envelope.area, 1e-9), 4)))
            draw.line(points + points[:1], fill=["#42baff", "#ff5b30", "#ffe755", "#28ef97"][source_cls], width=2)
            x, y = np.mean(points, axis=0)
            draw.text((x, y), str(i), fill="white", stroke_width=1, stroke_fill="black")
        image.thumbnail((380, 300))
        image.save(args.out / "views" / (key + ".jpg"), quality=91)
        return dict(key=key, filename=row["filename"], source_split="train", image_sha256=sha(im),
                    published_manifest_sha256=row["sha256"], published_hash_matches=sha(im)==row["sha256"],
                    label_sha256=sha(label), image_url=ROOT+SUB+row["image_path"], label_url=ROOT+SUB+row["label_path"],
                    size=[w, h], source_pickup_indices=[r["instance_index"] for r in pickups if r["label_file"] == row["label_path"]],
                    objects=objects)

    with ThreadPoolExecutor(5) as pool:
        records = list(pool.map(stage, enumerate(rows)))
    (args.out / "index.json").write_text(json.dumps(dict(source=SOURCE, revision=REV, license="CC BY 4.0",
            manifest_sha256=sha(manifest), license_sha256=sha(license_text), records=records), indent=1))
    print(len(records), Counter(o["n"] for r in records for o in r["objects"]).most_common(6))


if __name__ == "__main__":
    main()
