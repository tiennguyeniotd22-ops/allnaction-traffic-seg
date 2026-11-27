"""Render original BDD training frames and labeled objects for manual instance review."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from overnight_drafts import COLORS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detailed = args.out / "detail"
    detailed.mkdir(exist_ok=True)
    records = []
    with ZipFile(args.dataset) as z:
        names = sorted(n for n in z.namelist() if n.startswith("images/train/bdd_") and n.endswith(".jpg"))
        assert len(names) == 180
        for i, name in enumerate(names):
            raw = z.read(name)
            image = Image.open(BytesIO(raw)).convert("RGB")
            w, h = image.size
            label = "labels/train/" + Path(name).stem + ".txt"
            body = z.read(label)
            lines = body.decode().splitlines()
            overlay = image.copy()
            pen = ImageDraw.Draw(overlay)
            items = []
            for j, l in enumerate(lines):
                fields = l.split()
                cls = int(fields[0])
                pts = np.asarray(fields[1:],dtype=float).reshape(-1,2) * [w,h]
                coords = list(map(tuple,pts))
                pen.line(coords+[coords[0]],fill=COLORS[cls],width=2)
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                pen.text(tuple(lo),f"{j}/{cls}",fill="white",stroke_width=1,stroke_fill="black")
                items.append(dict(row=j,cls=cls,box=[*lo.tolist(),*hi.tolist()]))
            records.append(dict(image=name,label=label,size=[w,h],image_sha256=hashlib.sha256(raw).hexdigest(),
                                label_sha256=hashlib.sha256(body).hexdigest(),objects=items))
            overlay.save(detailed/f"{i:03}.jpg",quality=95)
            if i % 6 == 0:
                canvas = Image.new("RGB",(3*385,2*460),"#ececec")
                caption = ImageDraw.Draw(canvas)
            x,y=(i%3)*385,(i%6//3)*460
            for im,dy in [(image,0),(overlay,218)]:
                im.thumbnail((382,216))
                canvas.paste(im,(x,y+dy))
            caption.text((x+2,y+435),f"{i:03} {Path(name).stem}  n={len(items)}",fill="black")
            if i%6==5:
                canvas.save(args.out / f"{i-5:03}.jpg",quality=92)
    (args.out/"index.json").write_text(json.dumps(records,indent=1)+"\n")
    print(len(records),sum(len(r["objects"]) for r in records),flush=True)


if __name__ == "__main__":
    main()
