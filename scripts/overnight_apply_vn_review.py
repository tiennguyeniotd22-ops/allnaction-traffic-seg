"""Apply pixel-reviewed edits to original team training annotations and verify all ZIP members."""
import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image

from overnight_dataset import encode, read, sha, write_member


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def edit(archive, record):
    name = record["image"]
    image_path = Path(name)
    assert image_path.parent.as_posix() == "images/train" and image_path.suffix == ".jpg"
    if re.fullmatch(r"zz_plus_cc\d{12}\.jpg", image_path.name):
        assert record["source"] == "COCO train2017"
        assert image_path.stem == f"zz_plus_cc{record['coco_image_id']:012d}"
    elif not image_path.name.startswith(("vn_sign_", "bdd_")):
        assert re.fullmatch(r"zz_drive_(?:syn_(?:rain|night)_)?\d{12}\.jpg", image_path.name)
        assert record["source"] == "COCO val2017"
    label = "labels/train/" + Path(name).stem + ".txt"
    data = archive.read(name)
    old = archive.read(label)
    assert sha(data) == record["image_sha256"] and sha(old) == record["label_sha256"]
    size = Image.open(BytesIO(data)).size
    assert list(size) == record["size"]
    lines = old.decode().splitlines()
    changed = {}
    extra = []
    for action in record["actions"]:
        op = action["op"]
        if op in {"append", "replace"}:
            serialized = encode(action["cls"], np.asarray(action["pixels"]), *size)
        if op == "append":
            extra.append(serialized)
            continue
        i = action["row"]
        assert 0 <= i < len(lines) and i not in changed
        cls, coords = lines[i].split(maxsplit=1)
        assert int(cls) == action["from_cls"]
        if "old_row_sha256" in action:
            assert sha(lines[i].encode()) == action["old_row_sha256"]
        if op == "relabel":
            assert 0 <= action["cls"] < 11 and action["cls"] != int(cls)
            changed[i] = f"{action['cls']} {coords}"
        elif op == "delete":
            changed[i] = None
        elif op == "replace":
            changed[i] = serialized
        else:
            raise ValueError(op)
    final = [changed.get(i, line) for i, line in enumerate(lines)]
    final = [line for line in final if line is not None] + extra
    assert final and len(final) == len(set(final))
    return label, ("\n".join(final) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "decisions", "validation", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    decisions = read(args.decisions)
    rows = decisions["records"]
    base_hash = file_sha(args.base)
    if "base_sha256" in decisions and decisions["base_sha256"] != base_hash:
        raise ValueError("Review was prepared for a different parent archive")
    val = read(args.validation)
    assert len(val) == 140 and len({r["file"] for r in val}) == 140
    args.out.mkdir(exist_ok=True, parents=True)
    target = args.out / "dataset.zip"
    temp = target.with_suffix(".partial")
    swaps = {}
    per_class = Counter()
    n_train = 0
    with ZipFile(args.base) as base:
        names = base.namelist()
        assert len(names) == len(set(names))
        for record in rows:
            name, body = edit(base, record)
            assert name not in swaps and body != base.read(name)
            swaps[name] = body
        with ZipFile(temp, "w") as out:
            for name in names:
                write_member(out, name, swaps[name] if name in swaps else base.read(name))
        with ZipFile(temp) as out:
            assert out.namelist() == names and out.testzip() is None
            for name in names:
                body = out.read(name)
                assert body == (swaps[name] if name in swaps else base.read(name))
                if name.startswith("labels/train/") and name.endswith(".txt"):
                    n_train += 1
                    per_class.update(int(line.split()[0]) for line in body.splitlines())
            for r in val:
                assert sha(out.read("images/val/" + r["file"])) == r["image_sha256"]
                assert sha(out.read("labels/val/" + Path(r["file"]).stem + ".txt")) == r["labels_sha256"]
    digest = file_sha(temp)
    if target.exists() and file_sha(target) != digest:
        raise FileExistsError(target)
    temp.replace(target)
    summary = dict(dataset_sha256=digest, base_sha256=base_hash, decisions_sha256=file_sha(args.decisions),
                   validation_manifest_sha256=file_sha(args.validation), checked_val_pairs=len(val),
                   train=n_train, changed_train_annotations=len(rows),
                   operations=Counter(a["op"] for r in rows for a in r["actions"]),
                   instances_by_class=[per_class[c] for c in range(11)],
                   changed_members={n: sha(b) for n, b in swaps.items()})
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(digest, n_train, len(swaps), summary["operations"], flush=True)


if __name__ == "__main__":
    main()
