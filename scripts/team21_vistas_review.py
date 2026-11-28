"""Prepare and index train-only Vistas candidates for an instance-by-instance visual review."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from overnight_dataset import sha
from overnight_drafts import COLORS
from overnight_extend import check_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    raw = args.root / "team21-mapillary-extra"
    converted = args.root / "team21-mapillary-extra-curated"
    out = args.root / "team21-vistas-review"
    for sub in ("images", "labels", "views", "sheets"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    index = json.loads((raw / "index.json").read_bytes())
    assert index["seed"] == 21 and index["excluded_parents"] == 482
    originals = {r["key"]: r for r in index["records"]}
    accepted = []
    for row in json.loads((converted / "manifest.json").read_bytes()):
        if "roi" not in row:
            continue
        key = row["key"]
        original = originals[key]
        assert original["source_split"] == "training" and original["image_sha256"] == row["source_sha256"]
        image_bytes = (converted / "images" / (key + ".jpg")).read_bytes()
        label_bytes = (converted / "labels" / (key + ".txt")).read_bytes()
        assert sha(image_bytes) == row["image_sha256"] and sha(label_bytes) == row["label_sha256"]
        w, h = row["size"]
        keep, labels, polygons = [], [], []
        threshold = max(original["width"], original["height"]) * .0125
        ambiguous_crop = False
        for obj, line in zip(row["objects"], label_bytes.decode().splitlines(), strict=True):
            points = np.array(line.split()[1:], float).reshape(-1, 2)
            short = (np.ptp(points, axis=0) * [w, h]).min()
            if short < threshold:
                if short >= max(w, h)*.0125:
                    ambiguous_crop = True
                continue
            keep.append(obj)
            labels.append(line)
            polygons.append(points)
        if ambiguous_crop or not 1 <= len(keep) <= 18 or min(o["iou"] for o in keep) < .95:
            continue
        if not any(o["cls"] in {1, 2, 4, 5, 6, 7, 9, 10} and o["area640"] >= 140 for o in keep):
            continue
        body = ("\n".join(labels) + "\n").encode()
        try:
            check_labels(body, keep)
        except ValueError:
            continue
        with Image.open(BytesIO(image_bytes)) as src:
            im = src.convert("RGB")
        im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=94, subsampling=0)
        jpg = buf.getvalue()
        (out / "images" / (key + ".jpg")).write_bytes(jpg)
        (out / "labels" / (key + ".txt")).write_bytes(body)
        size = list(im.size)
        im.thumbnail((640, 420), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(im)
        for poly, obj in zip(polygons, keep, strict=True):
            ring = list(map(tuple, poly * im.size))
            draw.line(ring + [ring[0]], fill=COLORS[obj["cls"]], width=2)
        im.save(out / "views" / (key + ".jpg"), quality=93)
        parent = raw / "images" / (key + ".jpg")
        accepted.append(dict(row, size=size, unscaled_size=[w,h], objects=keep, image_sha256=sha(jpg),
                             label_sha256=sha(body), dropped_below_short_side=len(row["objects"])-len(keep),
                             parent=parent.relative_to(args.root).as_posix(), parent_sha256=original["image_sha256"],
                             source=index["source"], url=index["homepage"], license=index["license"],
                             credit=index["credit"], license_url=index["license_url"], revision=index["revision"],
                             source_url=index["mirror"]+row["source_image"], mask_url=index["mirror"]+row["source_mask"]))
    accepted.sort(key=lambda r: (r["selection"], r["key"]))
    (out / "manifest.json").write_bytes(json.dumps(accepted, indent=1).encode())
    (out / "review-order.json").write_bytes(json.dumps([r["key"] for r in accepted]).encode())
    for start in range(0, len(accepted), 12):
        sheet = Image.new("RGB", (4*380, 3*240), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(accepted[start:start+12]):
            x, y = j%4*380, j//4*240
            with Image.open(out / "views" / (row["key"]+".jpg")) as view:
                view.thumbnail((377,220))
                sheet.paste(view,(x+(380-view.width)//2,y))
            draw.text((x+2,y+222), f"{start+j:03d} " + " ".join(str(o["cls"]) for o in row["objects"]), fill="black")
        sheet.save(out / "sheets" / f"{start:03d}.jpg", quality=93)
    print(len(accepted), Counter(r["selection"] for r in accepted), flush=True)


if __name__ == "__main__":
    main()
