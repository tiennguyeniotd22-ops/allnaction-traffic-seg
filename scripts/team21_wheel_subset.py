"""Construct a train split focused on ambiguous two-wheelers. Validate every frozen val pair."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha(body):
    return hashlib.sha256(body).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "validation", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    val = json.loads(args.validation.read_bytes())
    assert len(val) == 140 and sha(args.validation.read_bytes()) == "23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013"
    args.out.mkdir(exist_ok=True, parents=True)
    output = args.out / "dataset.zip"
    assert not output.exists()
    tmp = output.with_suffix(".partial")
    keep = set()
    counts = Counter()
    with ZipFile(args.base) as src, ZipFile(tmp, "w") as dst:
        for name in sorted(src.namelist()):
            if not name.startswith("labels/train/") or not name.endswith(".txt"):
                continue
            classes = {int(line.split()[0]) for line in src.read(name).splitlines() if line.strip()}
            stem = Path(name).stem
            target = bool(classes & {4, 6, 7})
            probability = 0.65 if 5 in classes else .22
            rank = int(sha(("21/wheel/"+stem).encode())[:12],16)/(2**48)
            if target or rank < probability:
                keep.add(stem)
                counts.update(classes)
        for info in src.infolist():
            if "/train/" in info.filename and Path(info.filename).stem not in keep:
                continue
            dst.writestr(info,src.read(info))
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for row in val:
            name = row["file"]
            assert sha(z.read("images/val/"+name)) == row["image_sha256"]
            assert sha(z.read("labels/val/"+Path(name).stem+".txt")) == row["labels_sha256"]
        hashes = {sha(z.read(n)) for n in z.namelist() if n.startswith("images/train/")}
        assert len(hashes)==len(keep) and hashes.isdisjoint(row["image_sha256"] for row in val)
    tmp.replace(output)
    manifest = dict(train=len(keep),val=len(val),seed=21,classes=[4,6,7],counts=counts,
                    base_sha256=sha(args.base.read_bytes()),dataset_sha256=sha(output.read_bytes()),
                    validation_manifest_sha256=sha(args.validation.read_bytes()),train_stems=sorted(keep))
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=1)+"\n")
    print(manifest["train"], manifest["counts"], manifest["dataset_sha256"], flush=True)


if __name__=="__main__":
    main()
