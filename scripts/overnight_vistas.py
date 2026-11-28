"""Stage licensed Vistas v2 training images paired with native panoptic IDs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
from urllib.request import urlopen

import requests


REPO = "ali-sh07/Mapillary-dataset"
REV = "16ce879b802c3873a1a9e0da4113de0db1b522bf"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
HOME = "https://www.mapillary.com/dataset/vistas"
INDEX = "training/v2.0/panoptic/panoptic_2020.json"
INDEX_SHA = "b3b0e1080b2e45e7a637924b8520faa0a242a6eb19d9c192430a04f1430ac4d2"


def listing(folder):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{folder}?limit=1000"
    entries = {}
    while url:
        response = requests.get(url, timeout=35)
        response.raise_for_status()
        for item in response.json():
            entries[item["path"]] = (item["size"], item.get("lfs", {}).get("oid", item.get("oid")))
        url = response.links.get("next", {}).get("url")
    return entries


def download(name, dest, expected=None):
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        for attempt in range(3):
            try:
                with urlopen(BASE + name, timeout=40) as remote, tmp.open("wb") as f:
                    shutil.copyfileobj(remote, f)
                tmp.replace(dest)
                break
            except OSError:
                if attempt == 2:
                    raise
    content = dest.read_bytes()
    digest = sha256(content).hexdigest()
    if expected:
        size, oid = expected
        if len(oid) == 40:
            import hashlib
            object_digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        else:
            object_digest = digest
        if size != len(content) or oid != object_digest:
            raise ValueError(f"Mirror object hash mismatch for {name}")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-bucket", type=int, default=36)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--exclude-index", action="append", type=Path, default=[], help="Exclude every parent ID in a prior collection")
    parser.add_argument("--trucks", action="store_true", help="Also collect trucks to review against buses and pickups")
    args = parser.parse_args()
    folder = args.out
    for d in ("images", "native"):
        (folder / d).mkdir(parents=True, exist_ok=True)
    assert download(INDEX, folder / "panoptic_2020.json") == INDEX_SHA
    for name in ("LICENSE", "README", "config_v2.0.json"):
        download(name, folder / name)
    assert "CC BY-NC-SA" in (folder / "LICENSE").read_text()
    info = json.loads((folder / "panoptic_2020.json").read_bytes())
    categories = {c["id"]: c["name"] for c in info["categories"]}
    files = {i["id"]: i for i in info["images"]}
    with ThreadPoolExecutor(2) as pool:
        images, masks = pool.map(listing, ["training/images", "training/v2.0/panoptic"])
    pools = defaultdict(list)
    records = {}
    for a in info["annotations"]:
        i = files[a["image_id"]]
        image = "training/images/" + i["file_name"]
        mask = "training/v2.0/panoptic/" + a["file_name"]
        if image not in images or mask not in masks:
            continue
        if not .75 <= i["width"] / i["height"] <= 2.8 or i["width"] * i["height"] > 12500000:
            continue
        key = a["image_id"]
        usable = {categories[s["category_id"]].split("--")[-1] for s in a["segments_info"]
                  if not s["iscrowd"] and s["area"] * (640 / max(i["width"], i["height"]))**2 >= 120}
        buckets = usable & {"bicyclist", "motorcyclist", "trailer", "bus", "general-single", "direction-front"}
        if args.trucks and "truck" in usable:
            buckets.add("truck")
        for vehicle, rider in [("motorcycle", "motorcyclist"), ("bicycle", "bicyclist")]:
            if vehicle in usable and rider not in usable:
                buckets.add(vehicle)
        for b in buckets:
            pools[b].append(key)
        records[key] = dict(key="vista_" + key, width=i["width"], height=i["height"], source_split="training",
                            image=image, annotation=mask, segments=a["segments_info"])
    excluded = set()
    prior_hashes = []
    for index in args.exclude_index:
        content = index.read_bytes()
        prior = json.loads(content)
        assert prior["revision"] == REV
        excluded.update(r["key"].removeprefix("vista_") for r in prior["records"])
        prior_hashes.append(sha256(content).hexdigest())
    rng = random.Random(args.seed)
    chosen = {}
    for bucket in sorted(pools, key=lambda b: (len(pools[b]), b)):
        options = sorted(set(pools[bucket]) - chosen.keys() - excluded)
        rng.shuffle(options)
        chosen.update({k: bucket for k in options[:args.per_bucket]})

    def fetch(key):
        row = records[key]
        image, mask = row["image"], row["annotation"]
        return dict(row, selection=chosen[key],
                    image_sha256=download(image, folder / "images" / (row["key"] + ".jpg"), images[image]),
                    annotation_sha256=download(mask, folder / "native" / (row["key"] + ".png"), masks[mask]))

    with ThreadPoolExecutor(6) as pool:
        picked = list(pool.map(fetch, sorted(chosen)))
    report = dict(source="Mapillary Vistas v2.0 research edition", homepage=HOME,
                  credit="Neuhold, Ollmann, Rota Bulò and Kontschieder, ICCV 2017; Mapillary Research",
                  license="CC BY-NC-SA 4.0", license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                  revision=REV, mirror=BASE, source_index_sha256=INDEX_SHA, categories=categories,
                  candidate_pool={k: len(v) for k, v in pools.items()}, records=picked,
                  seed=args.seed, excluded_parents=len(excluded),
                  excluded_index_sha256=prior_hashes[0] if len(prior_hashes)==1 else None)
    if len(prior_hashes)>1:
        report["excluded_indices_sha256"] = prior_hashes
    (folder / "index.json").write_text(json.dumps(report, indent=1))
    print(len(picked), Counter(r["selection"] for r in picked), flush=True)


if __name__ == "__main__":
    main()
