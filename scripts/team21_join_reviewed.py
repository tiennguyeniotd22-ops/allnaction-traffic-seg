"""Add reviewed BDD and Mapillary train crops to the Team 21 guideline-reviewed corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile, ZipInfo


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "validation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reviewed = args.root / "team21-road-v1/dataset.zip"
    wide = args.root / "e3-vnrepair-v1/dataset.zip"
    sources = {p.parent.name: digest(p) for p in (reviewed, wide)}
    assert sources == {
        "team21-road-v1": "155b1d3cc81ea8b87c6123795a02702eb847b3807c85b76d4d21e496225e3252",
        "e3-vnrepair-v1": "03b99ac7680044dfb2ae2b1e22839d78fcfc20cf853e6c33c39b2b4d5e52daa2",
    }
    val_sha = digest(args.validation)
    assert val_sha == "23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013"
    val = json.loads(args.validation.read_bytes())
    assert len(val) == 140
    excluded = {row["image_sha256"] for row in val}
    for path in (args.root / "e2-v1/holdout.zip", args.root / "probe-frozen.zip"):
        with ZipFile(path) as archive:
            excluded.update(sha(archive.read(k)) for k in archive.namelist() if k.startswith("images/") and k.endswith(".jpg"))
    counts = Counter()
    kept_labels = []
    added = []
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        output = Path(tmp) / "release"
        output.mkdir()
        target = output / "dataset.zip"
        shutil.copyfile(reviewed, target)
        with ZipFile(target, "a") as dst, ZipFile(wide) as donor:
            original, candidates = set(dst.namelist()), set(donor.namelist())
            assert len(dst.namelist()) == len(original) and len(donor.namelist()) == len(candidates)
            for name in sorted(original & candidates):
                if name.startswith("labels/train/"):
                    if dst.read(name) != donor.read(name):
                        kept_labels.append(name)
                else:
                    assert dst.read(name) == donor.read(name), name
            fresh = candidates - original
            images = sorted(n for n in fresh if n.startswith("images/train/") and n.endswith(".jpg"))
            assert len(images) == 255 and len(fresh) == 2 * len(images)
            for image in images:
                stem = Path(image).stem
                label = f"labels/train/{stem}.txt"
                assert label in fresh
                counts["Mapillary" if stem.startswith("zz_plus_vista_") else "BDD" if stem.startswith("zz_plus_bdsem_") else "unknown"] += 1
                for name in (image, label):
                    data = donor.read(name)
                    info = ZipInfo(name, (2026, 11, 27, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    dst.writestr(info, data, compress_type=ZIP_STORED)
                added.append(stem)
        assert counts == {"Mapillary": 212, "BDD": 43} and len(kept_labels) == 48
        with ZipFile(target) as archive:
            assert archive.testzip() is None
            for row in val:
                image = row["file"]
                assert sha(archive.read("images/val/" + image)) == row["image_sha256"]
                assert sha(archive.read("labels/val/" + Path(image).stem + ".txt")) == row["labels_sha256"]
            train_images = [n for n in archive.namelist() if n.startswith("images/train/") and n.endswith(".jpg")]
            train_shas = {sha(archive.read(n)) for n in train_images}
            assert len(train_images) == len(train_shas) == 1442 and not train_shas & excluded
        manifest = dict(dataset_sha256=digest(target), source_datasets=sources, train=len(train_images),
                        validation_manifest_sha256=val_sha, val=len(val), added_by_source=counts,
                        added_train_stems=added, retained_reviewed_labels=kept_labels)
        (output / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode())
        if args.output.exists():
            assert digest(args.output / "dataset.zip") == manifest["dataset_sha256"]
            assert json.loads((args.output / "manifest.json").read_bytes()) == manifest
        else:
            output.rename(args.output)
        print(manifest["dataset_sha256"], manifest["train"], dict(counts))


if __name__ == "__main__":
    main()
