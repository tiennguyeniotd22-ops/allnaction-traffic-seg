"""Append reviewed train examples and verify the frozen validation split on disk."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from overnight_dataset import read, sha, write_member


def check_labels(label, objects):
    lines = label.decode().splitlines()
    if not lines or len(lines) != len(set(lines)) or len(lines) != len(objects):
        raise ValueError("Empty, duplicate, or missing annotation")
    for line, obj in zip(lines, objects, strict=True):
        fields = line.split()
        points = np.asarray(fields[1:], float).reshape(-1, 2)
        if int(fields[0]) != obj["cls"] or not 0 <= obj["cls"] <= 10:
            raise ValueError("Inconsistent class")
        if not np.isfinite(points).all() or points.min() < 0 or points.max() > 1:
            raise ValueError("Invalid normalized coordinates")
        shape = Polygon(points)
        if not shape.is_valid or shape.area <= 0:
            raise ValueError("Invalid polygon")


def main():
    parser = argparse.ArgumentParser()
    for arg in ("base", "val", "folder", "selection", "scene_report", "out"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    config = read(args.selection)
    folder = args.folder
    records = {r["key"]: r for r in read(folder / "manifest.json")}
    keys = set(config["keep"])
    assert keys and len(keys) == len(config["keep"])
    assert sha((folder / "manifest.json").read_bytes()) == config["source_manifest_sha256"]
    scenes = read(args.scene_report)
    assert scenes["candidate_manifest_sha256"][folder.name] == config["source_manifest_sha256"]
    assert scenes["base_dataset_sha256"] == sha(args.base.read_bytes())
    for a, b in ([(e["a"], e["b"]) for e in scenes["same_scene"]] +
                 [(e[0], e[1]) for e in scenes["close_hash"]]):
        if a in keys and (b in keys or b == "base") or b in keys and a == "base":
            raise ValueError(f"Unresolved overlap: {a}, {b}")

    val = {r["file"]: r for r in read(args.val)}
    assert len(val) == 140
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "dataset.zip"
    if out.exists():
        raise FileExistsError(out)
    tmp = out.with_suffix(".partial")
    seen = set()
    counts = Counter()
    additions = []
    with ZipFile(args.base) as base, ZipFile(tmp, "w") as z:
        assert len(base.namelist()) == len(set(base.namelist()))
        for name in base.namelist():
            data = base.read(name)
            write_member(z, name, data)
            if name.startswith("images/"):
                seen.add(sha(data))
        for key in sorted(keys):
            row = records[key]
            image = (folder / "images" / (key + ".jpg")).read_bytes()
            label = (folder / "labels" / (key + ".txt")).read_bytes()
            assert sha(image) == row["image_sha256"] and sha(label) == row["label_sha256"]
            assert sha(image) not in seen and Image.open(BytesIO(image)).size == tuple(row["size"])
            check_labels(label, row["objects"])
            seen.add(sha(image))
            stem = "zz_plus_" + key
            assert "images/train/" + stem + ".jpg" not in z.NameToInfo
            write_member(z, "images/train/" + stem + ".jpg", image)
            write_member(z, "labels/train/" + stem + ".txt", label)
            counts.update(o["cls"] for o in row["objects"])
            additions.append(dict(row, stem=stem))
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for split in ("train", "val"):
            imgs = {Path(n).stem for n in z.namelist() if n.startswith(f"images/{split}/") and n.endswith(".jpg")}
            lbls = {Path(n).stem for n in z.namelist() if n.startswith(f"labels/{split}/") and n.endswith(".txt")}
            assert imgs == lbls
            if split == "val":
                assert imgs == {Path(n).stem for n in val}
                for name, row in val.items():
                    assert sha(z.read("images/val/" + name)) == row["image_sha256"]
                    assert sha(z.read("labels/val/" + Path(name).stem + ".txt")) == row["labels_sha256"]
            else:
                n_train = len(imgs)
    tmp.replace(out)
    summary = dict(train=n_train, val=len(val), added_images=len(keys), added_by_class=counts,
                   dataset_sha256=sha(out.read_bytes()), base_sha256=scenes["base_dataset_sha256"],
                   holdout_sha256=scenes["holdout_sha256"], scene_report_sha256=sha(args.scene_report.read_bytes()),
                   selection_sha256=sha(args.selection.read_bytes()))
    (args.out / "manifest.json").write_text(json.dumps(dict(summary, records=additions), indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
