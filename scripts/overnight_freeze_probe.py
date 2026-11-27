"""Build and lock an out-of-source diagnostic pack before model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from overnight_dataset import write_member


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser()
    for k in ("root", "decisions", "pack", "manifest"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    root = a.root
    decisions = read(a.decisions)
    candidates = root / "vistas-probe-candidates"
    scenes = read(root / "vistas-probe-scenes.json")
    assert scenes["candidate_manifest_sha256"][candidates.name] == digest(candidates / "manifest.json")
    assert scenes["base_dataset_sha256"] == digest(root / "e3-vistas-v1/dataset.zip")
    assert scenes["holdout_sha256"] == digest(root / "e2-v1/holdout.zip")
    order_file = root / "vistas-probe-curated/qc/order.json"
    order = read(order_file)
    assert decisions["order_sha256"] == digest(order_file)
    eligible = {order[i] for i in decisions["approved_indices"]}
    flagged = {m[k] for m in scenes["same_scene"] for k in ("a", "b")}
    flagged.update(k for m in scenes["close_hash"] for k in m[:2])
    rows = {r["key"]: r for r in read(candidates / "manifest.json")}
    prior = read(root / "vistas/index.json")
    source = read(root / "vistas-probe/index.json")
    assert source["excluded_index_sha256"] == digest(root / "vistas/index.json")
    assert not ({r["key"] for r in prior["records"]} & {r["key"] for r in source["records"]})
    assert eligible <= rows.keys()
    files = {}
    records = []
    counts = Counter()
    temp = a.pack.with_suffix(".partial")
    with ZipFile(temp, "w") as z:
        for key in sorted(eligible - flagged):
            row = rows[key]
            assert digest(root / row["parent"]) == row["parent_sha256"]
            assert digest(root / "vistas-probe/native" / (key + ".png")) == row["source_mask_sha256"]
            label = candidates / "labels" / (key + ".txt")
            classes = [int(line.split()[0]) for line in label.read_bytes().splitlines()]
            assert classes == [obj["cls"] for obj in row["objects"]]
            counts.update(classes)
            for folder, suffix, field in [("labels", ".txt", "label_sha256"), ("images", ".jpg", "image_sha256")]:
                member = f"{folder}/{key}{suffix}"
                path = candidates / member
                assert digest(path) == row[field]
                files[member] = row[field]
                write_member(z, member, path.read_bytes())
            records.append({k: row[k] for k in ("key", "roi", "selection", "parent_sha256", "source_mask_sha256", "source_url")})
    locked = dict(role="diagnostic only; never used for training or model selection",
                  zip_sha256=digest(temp), files=files, records=records,
                  source=source["homepage"], license=source["license"], revision=source["revision"],
                  raw_instances_by_class=[counts[c] for c in range(11)],
                  reviewed=len(order), approved_visual=len(eligible), after_scene_exclusion=len(records),
                  scene_report_sha256=digest(root / "vistas-probe-scenes.json"),
                  reference_train_sha256=scenes["base_dataset_sha256"], checked_class3_holdout=scenes["holdout_sha256"])
    if a.pack.exists() and digest(a.pack) != locked["zip_sha256"]:
        raise ValueError("Probe already frozen with different bytes")
    if a.manifest.exists() and read(a.manifest) != locked:
        raise ValueError("Frozen manifest differs")
    temp.replace(a.pack)
    a.manifest.write_text(json.dumps(locked, indent=1) + "\n", encoding="utf8")
    print(len(records), locked["raw_instances_by_class"], locked["zip_sha256"])


if __name__ == "__main__":
    main()
