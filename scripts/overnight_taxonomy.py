"""Apply image-reviewed class and duplicate decisions to a frozen train split."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from shapely import make_valid
from shapely.geometry import LineString, Polygon, box
from shapely.ops import nearest_points, unary_union

from overnight_dataset import encode, read, sha, write_member
from overnight_drafts import outline, pieces, to_rle


def repair_ring(coords, cls, width, height):
    pixels = coords * [width, height]
    native = to_rle(pixels, width, height)
    parts = pieces(make_valid(Polygon(pixels)))
    if not parts:
        return None
    parts.sort(key=lambda poly: -poly.area)
    # Buffer only along touching pieces or slits. Verify the final serialized ring.
    candidates = []
    for epsilon in (.008, .09, .35):
        merged = parts[0]
        for piece in parts[1:]:
            a, b = nearest_points(merged, piece)
            merged = unary_union([merged, piece, LineString([a, b]).buffer(epsilon)])
        for poly in pieces(merged.intersection(box(0, 0, width, height))):
            for hole in list(poly.interiors):
                a, b = nearest_points(hole, poly.exterior)
                cut = pieces(poly.difference(LineString([a, b]).buffer(epsilon)))
                if cut:
                    poly = max(cut, key=lambda g: g.area)
            candidates.append(np.asarray(poly.exterior.coords[:-1]))
    candidates.append(outline(make_valid(Polygon(pixels)), width, height))
    best = None
    for points in candidates:
        if points is None:
            continue
        try:
            line = encode(cls, points, width, height)
        except ValueError:
            continue
        serialized = np.asarray(line.split()[1:], float).reshape(-1, 2) * [width, height]
        fit = float(mu.iou([native], [to_rle(serialized, width, height)], [0])[0, 0])
        if fit >= .985 and (best is None or fit > best[0]):
            best = (fit, line)
        if fit == 1:
            break
    return best


def main():
    parser = argparse.ArgumentParser()
    for arg in ("base", "decisions", "val", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--repair-train-rings", action="store_true")
    args = parser.parse_args()
    decisions = read(args.decisions)
    assert sha(args.base.read_bytes()) == decisions["base_sha256"]
    validation = read(args.val)
    assert len(validation) == len({r["file"] for r in validation}) == 140
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "dataset.zip"
    if out.exists():
        raise FileExistsError(out)
    changes = {}
    log, repairs = [], []
    with ZipFile(args.base) as source:
        assert len(source.namelist()) == len(set(source.namelist()))
        for name, item in decisions["files"].items():
            assert name.startswith("labels/train/") and name.endswith(".txt")
            image = "images/train/" + Path(name).stem + ".jpg"
            data = source.read(name)
            assert sha(data) == item["label_sha256"] and sha(source.read(image)) == item["image_sha256"]
            if "exclude" in item:
                assert Path(name).stem.startswith("zz_plus_")
                changes[image] = changes[name] = None
                log.append(dict(file=name, before=sha(data), after=None, reason=item["exclude"]))
                continue
            lines = data.decode().splitlines()
            seen = set()
            for edit in item["changes"]:
                index = edit["line"]
                assert index not in seen and sha(lines[index].encode()) == edit["row_sha256"]
                seen.add(index)
                old, coords = lines[index].split(maxsplit=1)
                assert int(old) == edit["old_class"]
                target = edit["target"]
                assert target is None or 0 <= target <= 10
                lines[index] = None if target is None else f"{target} {coords}"
            kept = [line for line in lines if line is not None]
            assert kept and len(kept) == len(set(kept))
            result = ("\n".join(kept) + "\n").encode()
            changes[name] = result
            log.append(dict(file=name, before=sha(data), after=sha(result)))
        tmp = out.with_suffix(".partial")
        with ZipFile(tmp, "w") as dest:
            for name in source.namelist():
                data = changes[name] if name in changes else source.read(name)
                if data is not None and args.repair_train_rings and name.startswith("labels/train/"):
                    lines = data.decode().splitlines()
                    size = None
                    for i, row in enumerate(lines):
                        fields = row.split()
                        coords = np.asarray(fields[1:], float).reshape(-1, 2)
                        if Polygon(coords).is_valid:
                            continue
                        if size is None:
                            with Image.open(BytesIO(source.read("images/train/" + Path(name).stem + ".jpg"))) as im:
                                size = im.size
                        candidate = repair_ring(coords, int(fields[0]), *size)
                        if candidate:
                            fit, lines[i] = candidate
                            repairs.append(dict(file=name, line_after_taxonomy=i, iou=fit,
                                                before=sha(row.encode()), after=sha(lines[i].encode())))
                    if lines != data.decode().splitlines():
                        data = ("\n".join(lines) + "\n").encode()
                if data is not None:
                    write_member(dest, name, data)

    counts, instances = {}, {}
    invalid = []
    train_hashes = set()
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for split in ("train", "val"):
            imgs = {Path(n).stem for n in z.namelist() if n.startswith(f"images/{split}/")}
            labs = {Path(n).stem for n in z.namelist() if n.startswith(f"labels/{split}/")}
            assert imgs == labs
            counts[split] = len(imgs)
            cls_counts = Counter()
            for stem in sorted(imgs):
                digest = sha(z.read(f"images/{split}/{stem}.jpg"))
                if split == "train":
                    train_hashes.add(digest)
                else:
                    assert digest not in train_hashes
                for line, row in enumerate(z.read(f"labels/{split}/{stem}.txt").decode().splitlines()):
                    col = row.split()
                    cls = int(col[0])
                    coords = np.array(col[1:], float).reshape(-1, 2)
                    assert 0 <= cls <= 10 and len(coords) >= 3
                    assert np.isfinite(coords).all() and coords.min() >= 0 and coords.max() <= 1
                    polygon = Polygon(coords)
                    if not polygon.is_valid or polygon.area <= 0:
                        invalid.append(dict(split=split, stem=stem, line=line))
                    cls_counts[cls] += 1
            instances[split] = dict(sorted(cls_counts.items()))
        assert counts["val"] == 140
        for row in validation:
            name = row["file"]
            assert sha(z.read("images/val/" + name)) == row["image_sha256"]
            assert sha(z.read("labels/val/" + Path(name).stem + ".txt")) == row["labels_sha256"]
    tmp.replace(out)
    totals = dict(counts=counts, instances=instances, base_sha256=decisions["base_sha256"],
                  dataset_sha256=sha(out.read_bytes()), decisions_sha256=sha(args.decisions.read_bytes()),
                  validation_manifest_sha256=sha(args.val.read_bytes()), invalid_geometry=invalid)
    (args.out / "manifest.json").write_text(json.dumps(dict(totals, taxonomy_edits=log, ring_repairs=repairs), indent=1))
    print(json.dumps(dict(totals, invalid_geometry=len(invalid), repaired_train=len(repairs))))


if __name__ == "__main__":
    main()
