"""Select a reproducible 40+10 train-only audit and render original frames and annotations."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from overnight_apply_vn_review import file_sha
from overnight_collect_coco import dhash
from overnight_dataset import read, sha
from overnight_drafts import COLORS


def main():
    p = argparse.ArgumentParser()
    for name in ("dataset", "prior", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    rng = random.Random(21)
    archive_sha = file_sha(a.dataset)
    if archive_sha != "33362fd1aee1760da38f0f67e41a97c61dcc6aed6c10561bfbf9ef058755c2ef":
        p.error("Expected the team21-qc20 pilot dataset")
    old = {Path(k).stem for k in read(a.prior)["changed_members"]}
    a.out.mkdir(exist_ok=True, parents=True)
    with ZipFile(a.dataset) as z:
        files = sorted(n for n in z.namelist() if n.startswith("images/train/") and n.endswith(".jpg") and "_syn_" not in n)
        labels = {n: z.read("labels/train/" + Path(n).stem + ".txt") for n in files}
        classes = {n: {int(line.split()[0]) for line in body.splitlines()} for n, body in labels.items()}
        selected, fingerprints = [], []

        def choose(group, n_images, eligible):
            pool = sorted(n for n in files if n not in {r["image"] for r in selected} and eligible(n))
            rng.shuffle(pool)
            count = 0
            for n in pool:
                body = z.read(n)
                im = Image.open(BytesIO(body))
                fp = dhash(im)
                if any((fp ^ p).bit_count() <= 6 for p in fingerprints):
                    continue
                selected.append(dict(index=len(selected), group=group, image=n, size=list(im.size),
                                     image_sha256=sha(body), label_sha256=sha(labels[n])))
                fingerprints.append(fp)
                count += 1
                if count == n_images:
                    break
            if count != n_images:
                raise ValueError((group, count, n_images))

        for group, ids, strata in [
            ("two_wheels", {4, 5, 6, 7}, [("zz_drive_", 11), ("vn_sign_", 6), ("zz_plus_cc", 3)]),
            ("heavy_vehicle", {2, 3}, [("bdd_", 4), ("zz_drive_", 5), ("vn_sign_", 1)]),
            ("sign_or_light", {9, 10}, [("bdd_", 3), ("vn_sign_", 4), ("zz_drive_", 3)]),
        ]:
            for prefix, count in strata:
                choose(group, count, lambda n: Path(n).name.startswith(prefix) and Path(n).stem not in old and classes[n] & ids)
        choose("random", 10, lambda n: True)
        manifest = dict(dataset_sha256=archive_sha, seed=21, groups=Counter(r["group"] for r in selected),
                        population=len(files), records=selected)
        out = a.out / "sample.json"
        if out.exists() and read(out) != manifest:
            raise ValueError("Stored review sample differs")
        out.write_text(json.dumps(manifest, indent=1) + "\n")
        for r in selected:
            idx, n = r["index"], r["image"]
            im = Image.open(BytesIO(z.read(n))).convert("RGB")
            view = im.copy()
            draw = ImageDraw.Draw(view)
            wh = np.array(im.size)
            for i, line in enumerate(labels[n].splitlines()):
                cls, *coords = line.split()
                cls = int(cls)
                pts = [tuple(p) for p in np.array(coords, float).reshape(-1, 2) * wh]
                draw.line(pts + pts[:1], fill=COLORS[cls], width=2)
                x, y = np.min(pts, axis=0)
                draw.text((x, y), f"{i}/{cls}", fill=COLORS[cls], stroke_width=1, stroke_fill="black")
            for kind, frame in (("raw", im), ("overlay", view)):
                target = a.out / kind
                target.mkdir(exist_ok=True)
                frame.save(target / f"{idx:02}.jpg", quality=96)
        for kind in ("raw", "overlay"):
            for i in range(0, len(selected), 4):
                page = Image.new("RGB", (1200, 860), "#eeeeee")
                pen = ImageDraw.Draw(page)
                for j, r in enumerate(selected[i:i+4]):
                    im = Image.open(a.out / kind / f"{r['index']:02}.jpg")
                    im.thumbnail((595, 397))
                    x, y = (j % 2)*600, (j // 2)*430
                    page.paste(im,(x,y))
                    pen.text((x+2,y+401),f"{r['index']:02}  {Path(r['image']).name}   {r['group']}",fill="black")
                page.save(a.out / f"{kind}-{i:02}.jpg",quality=94)
        print(out, manifest["groups"], flush=True)


if __name__ == "__main__":
    main()
