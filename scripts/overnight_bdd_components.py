"""Stage official BDD train semantic regions for visual instance separation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from urllib.request import urlopen

import numpy as np
from PIL import Image

from overnight_source_archive import SourceArchive


SOURCE = "https://www.kaggle.com/api/v1/datasets/download/alesssaulea/bdd100k?datasetVersionNumber=7"
LABEL_SPEC = "https://raw.githubusercontent.com/bdd100k/bdd100k/master/bdd100k/label/label.py"
LICENSE = "https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--per-bucket", type=int, default=45)
    args = p.parse_args()
    folder = args.out
    for sub in ("images", "native"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    ar = SourceArchive(SOURCE)
    labels = sorted(n for n in ar.archive.namelist() if n.startswith("labels/train/") and n.endswith("_train_id.png"))
    assert len(labels) == 7000
    # All train label members in this release fit into one contiguous 63 MB range.
    ordered = sorted((ar.archive.getinfo(n) for n in labels), key=lambda info: info.header_offset)
    start = ordered[0].header_offset
    end = ordered[-1].header_offset + ordered[-1].compress_size + 1024
    assert end - start < 80 * 2**20
    block = start, ar.fetch(start, end)
    with urlopen(LABEL_SPEC, timeout=20) as r:
        spec = r.read()
    with urlopen(LICENSE, timeout=20) as r:
        license_text = r.read()
    (folder / "license.rst").write_bytes(license_text)
    # Check the original dataset's label names and train IDs against this conversion.
    for name, original, train_id in [("motorcycle",37,17),("bicycle",33,18),("rider",32,12),
                                     ("bus",34,15),("traffic light",25,6),("traffic sign",26,7)]:
        assert f'Label("{name}", {original}, {train_id},'.encode() in spec
    bins = defaultdict(list)
    rng = random.Random(26)
    for name in labels:
        source = ar.member(name, block=block)
        with Image.open(BytesIO(source)) as im:
            assert im.mode == "L" and im.size == (1280, 720)
            sample = np.asarray(im)[::4, ::4]
        counts = np.bincount(sample.ravel(), minlength=256)
        for cls in [17,18,15,6,7]:
            if counts[cls] < (35 if cls in (17,18,15) else 14):
                continue
            bucket = str(cls)
            if cls in (17,18):
                bucket += "-rider" if counts[12] >= 12 else "-empty"
            bins[bucket].append(name)
    picked = {}
    for bucket, names in sorted(bins.items(), key=lambda kv: len(kv[1])):
        rng.shuffle(names)
        eligible = [n for n in names if n not in picked]
        picked.update({n: bucket for n in eligible[:args.per_bucket]})

    def fetch(pair):
        label_name, bucket = pair
        stem = Path(label_name).name.removesuffix("_train_id.png")
        key = "bdsem_" + stem
        image_name = "images/train/" + stem + ".jpg"
        rgb = ar.member(image_name)
        mask = ar.member(label_name, block=block)
        with Image.open(BytesIO(rgb)) as im:
            assert im.size == (1280, 720)
        (folder / "images" / (key + ".jpg")).write_bytes(rgb)
        (folder / "native" / (key + ".png")).write_bytes(mask)
        return dict(key=key, source_split="train", bucket=bucket, image=image_name, annotation=label_name,
                    image_sha256=sha256(rgb).hexdigest(), annotation_sha256=sha256(mask).hexdigest())

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(fetch, sorted(picked.items())))
    report = dict(source="BDD100K 10K semantic train", homepage="https://www.bdd100k.com/",
                  license="Berkeley DeepDrive dataset license for educational, research and nonprofit use",
                  version_url=SOURCE, label_spec=LABEL_SPEC, license_url=LICENSE,
                  license_sha256=sha256(license_text).hexdigest(), spec_sha256=sha256(spec).hexdigest(),
                  source_type="semantic train IDs; connected components require human inspection for instance boundaries",
                  candidate_pool={k:len(v) for k,v in bins.items()}, records=rows)
    (folder / "index.json").write_text(json.dumps(report, indent=1))
    print(len(rows), Counter(r["bucket"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
