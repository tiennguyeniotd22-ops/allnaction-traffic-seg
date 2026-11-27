"""Stage original train frames, native polygons and source IDs for the IDD experiment."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

from overnight_source_archive import SourceArchive


URL = "https://www.kaggle.com/api/v1/datasets/download/sounakp/idd-segmentation?datasetVersionNumber=1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--frames", type=int, default=720)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    source = SourceArchive(URL)
    groups = defaultdict(list)
    for member in source.archive.namelist():
        path = PurePosixPath(member)
        if "/gtFine/train/" in member and member.endswith("_gtFine_polygons.json"):
            groups[path.parent.name].append(member)
    seed = random.Random(26)
    drives = sorted(groups)
    seed.shuffle(drives)
    reserved = set(drives[:8])
    selected = []
    per_drive = {}
    for drive in drives:
        items = sorted(groups[drive])
        seed.shuffle(items)
        kept, indices = [], []
        for path in items:
            frame = int(PurePosixPath(path).name.split("_")[0])
            if all(abs(frame - old) >= 300 for old in indices):
                kept.append(path)
                indices.append(frame)
        per_drive[drive] = kept
    for i in range(max(map(len, per_drive.values()))):
        selected.extend((drive, per_drive[drive][i]) for drive in drives if len(per_drive[drive]) > i)
        if len(selected) >= a.frames:
            break
    for sub in ("images", "native"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    def get(pair):
        drive, name = pair
        image = name.replace("/gtFine/", "/leftImg8bit/").replace("_gtFine_polygons.json", "_leftImg8bit.png")
        key = "idd_" + drive + "_" + PurePosixPath(name).name.split("_")[0]
        summary = dict(key=key, drive=drive, annotation=name, image=image, split="train",
                       role="held_drive" if drive in reserved else "train_candidate")
        for field, path, output in (("annotation", name, a.out / "native" / (key+".json")),
                                    ("image", image, a.out / "images" / (key+".png"))):
            info = source.archive.getinfo(path)
            if output.is_file():
                body = output.read_bytes()
                import zlib
                if len(body) != info.file_size or zlib.crc32(body) != info.CRC:
                    raise IOError(f"Staged data differs from source for {key}")
            else:
                body = source.member(path)
                partial = output.with_suffix(".partial")
                partial.write_bytes(body)
                partial.replace(output)
            summary[field + "_sha256"] = hashlib.sha256(body).hexdigest()
        return summary

    items = []
    with ThreadPoolExecutor(a.workers) as pool:
        for row in pool.map(get, selected[:a.frames]):
            items.append(row)
            if len(items) % 25 == 0:
                print(len(items), flush=True)
    report = dict(source="India Driving Dataset, Segmentation (IDD 20k part I)",
                  author="IIIT Hyderabad and collaborators", original="https://idd.insaan.iiit.ac.in/dataset/details/",
                  mirror=URL, mirror_version=1, original_split="train", seed=26, minimum_frame_gap=300,
                  reserved_drives=sorted(reserved), records=items)
    (a.out / "index.json").write_text(json.dumps(report, indent=1))
    print("complete", len(items), len(groups), flush=True)


if __name__ == "__main__":
    main()
