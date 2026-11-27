"""Stage original Vietnamese training scenes and HCMC sign photos for polygon review."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageOps

from overnight_source_archive import SourceArchive


SOURCES = {
    "cam": ("huanhuyle/vietnam-traffic-detection", 2),
    "sign": ("nhduynguyn/traffic-sign-in-ho-chi-minh-city", 4),
}


def select(names, kind):
    rng = random.Random(26)
    groups = defaultdict(dict)
    for n in names:
        if kind == "cam":
            match = re.fullmatch(r"images/train/((?:day|night)_(?:cam|src)_\d+)_(?:frame_)?(\d+)(?:_jpg\.rf\.\w+)?\.jpg", n)
            if match:
                group, index = match.groups()
                if n.replace("images/", "labels/")[:-4] + ".txt" in names:
                    groups[group].setdefault(int(index), n)
        elif n.startswith("traffic_sign/traffic_sign/") and n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups[Path(n).parent.name][n] = n
    selected = []
    for group in sorted(groups):
        choices = sorted(groups[group])
        rng.shuffle(choices)
        kept = []
        limit = 4 if kind == "cam" else 38
        for key in choices:
            if kind == "cam" and any(abs(key - k) < 220 for k in kept):
                continue
            kept.append(key)
            selected.append((group, groups[group][key]))
            if len(kept) == limit:
                break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=SOURCES)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    slug, version = SOURCES[args.kind]
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}?datasetVersionNumber={version}"
    archive = SourceArchive(url)
    selected = select(set(archive.archive.namelist()), args.kind)
    directory = args.out / "images"
    directory.mkdir(parents=True, exist_ok=True)

    def stage(pair):
        group, name = pair
        body = archive.member(name)
        sha = hashlib.sha256(body).hexdigest()
        key = args.kind + "_" + sha[:15]
        with Image.open(BytesIO(body)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            original_size = image.size
            image.thumbnail((1440, 1440), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=94, subsampling=0)
        content = buffer.getvalue()
        (directory / (key + ".jpg")).write_bytes(content)
        item = dict(key=key, source=name, group=group, source_sha256=sha, original_size=original_size,
                    size=image.size, sha256=hashlib.sha256(content).hexdigest(), source_split="train" if args.kind=="cam" else "unsplit original photos")
        if args.kind == "cam":
            path = name.replace("images/", "labels/")[:-4] + ".txt"
            labels = archive.member(path)
            item.update(native_sha256=hashlib.sha256(labels).hexdigest(), boxes=[[float(v) for v in line.split()] for line in labels.decode().splitlines() if line.strip()])
        return item

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(stage, selected))
    unique = {r["key"]: r for r in rows}
    rows = sorted(unique.values(), key=lambda r: (r["group"], r["source"]))
    result = dict(dataset=slug, version=version, url=url, source_license="Apache 2.0 (dataset publisher's version metadata)",
                  seed=26, records=rows)
    (args.out / "index.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    print(args.kind, len(rows), "from", len({r["group"] for r in rows}), "groups", flush=True)


if __name__ == "__main__":
    main()
