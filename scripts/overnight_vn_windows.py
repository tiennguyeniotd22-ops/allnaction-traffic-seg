"""Build candidate Vietnam windows from source/SAM masks for visual acceptance."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from shapely.geometry import Polygon

from overnight_dataset import encode, sha
from overnight_drafts import draw, to_rle
from overnight_native_crops import polygon


def window(bounds, size, factor, min_side):
    x, y, right, bottom = bounds
    width, height = size
    side = max(min_side, max(right-x, bottom-y) * factor)
    cw = min(width, round(side * 1.22))
    ch = min(height, round(side))
    left = max(0, min(width-cw, round((x+right-cw)/2)))
    top = max(0, min(height-ch, round((y+bottom-ch)/2)))
    return left, top, left+cw, top+ch


def build(root, kind, prompt_allow):
    out = root / ("vn-"+kind+"-curated")
    for sub in ("images", "labels", "views"):
        (out/sub).mkdir(parents=True, exist_ok=True)
    index = json.loads((root/("vn-"+kind)/"index.json").read_text())
    source_meta = {row["key"]:row for row in index["records"]}
    with ZipFile(root/f"vn-{kind}-prompts.zip") as original, ZipFile(root/f"vn-{kind}-sam.zip") as predictions:
        assert original.testzip() is None and predictions.testzip() is None
        keys = {r["file"]:r["source_key"] for r in json.loads(original.read("manifest.json"))["records"]}
        prompts = json.loads(predictions.read("source-objects.json"))
        manifest = []
        labels = {"sign_proposal":9,"car":0,"bus":1,"truck":2,"moto_or_rider":5}
        for name, rows in sorted(prompts.items()):
            key = Path(name).stem
            ids = prompt_allow.get(str(int(key[1:])), []) if kind=="sign" else [r["index"] for r in rows]
            if not ids:
                continue
            data = original.read("images/"+name)
            frame = Image.open(BytesIO(data)).convert("RGB")
            meta = source_meta[keys[name]]
            assert sha(data)==meta["sha256"]
            objects = []
            for row in rows:
                idx = row["index"]
                if idx not in ids:
                    continue
                mask = np.asarray(Image.open(BytesIO(predictions.read(row["mask"]))).convert("L"))>0
                ys,xs = np.where(mask)
                if not len(xs):
                    continue
                bb = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                objects.append((idx, labels[row["source_label"]], mask, bb, row["quality"]))
            if kind=="sign":
                centers = objects[:1]
            else:
                centers = sorted(objects, key=lambda o: (o[4]>=.86, np.count_nonzero(o[2])),reverse=True)[:2]
            for idx, _, _, bb, _ in centers:
                # Context is presented along with every intersecting mask for final review.
                roi = window(bb, frame.size, 3.5 if kind=="cam" else 3, 210 if kind=="cam" else 185)
                l,t,r,b = roi
                crop = frame.crop(roi)
                cw,ch = crop.size
                lines, visible = [], []
                for obj,cls,mask,_,quality in objects:
                    region = np.ascontiguousarray(mask[t:b,l:r])
                    area = int(region.sum())*(640/max(cw,ch))**2
                    if area<1:
                        continue
                    pts = polygon(region)
                    if pts is None:
                        if area>=24:break
                        continue
                    try:
                        line = encode(cls,pts,cw,ch)
                    except ValueError:
                        if area>=24:break
                        continue
                    after = to_rle(np.asarray(line.split()[1:],float).reshape(-1,2)*[cw,ch],cw,ch)
                    original_mask = mu.encode(np.asfortranarray(region,dtype=np.uint8))
                    fit = float(mu.iou([after],[original_mask],[0])[0,0])
                    if fit < (.94 if area>=64 else .78):
                        if area>=24:break
                        continue
                    lines.append(line)
                    visible.append(dict(id=str(obj),cls=cls,area640=round(area,2),poly=pts.tolist(),iou=round(fit,4),sam_score=round(quality,4)))
                else:
                    if not visible:continue
                    ident = key+"p"+str(idx)
                    buf=BytesIO()
                    crop.save(buf,format="JPEG",quality=94,subsampling=0)
                    pixels=buf.getvalue()
                    label=("\n".join(lines)+"\n").encode()
                    (out/"images"/(ident+".jpg")).write_bytes(pixels)
                    (out/"labels"/(ident+".txt")).write_bytes(label)
                    draw(crop,visible).save(out/"views"/(ident+".jpg"),quality=93)
                    manifest.append(dict(key=ident,source=meta["source"],group=meta["group"],source_key=meta["key"],
                                         parent=key,source_sha256=meta["source_sha256"],prepared_parent_sha256=meta["sha256"],
                                         image_sha256=sha(pixels),label_sha256=sha(label),roi=roi,size=crop.size,
                                         objects=[{k:v for k,v in o.items() if k!="poly"} for o in visible]))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=1))
    print(kind,len(manifest),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("kind",choices=("cam","sign"))
    parser.add_argument("decisions",type=Path)
    args=parser.parse_args()
    build(args.root,args.kind,json.loads(args.decisions.read_text())["prompts"])
