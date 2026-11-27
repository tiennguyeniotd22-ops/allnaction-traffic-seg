"""Apply reviewed Vietnam crops and per-frame source taxonomy corrections."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from rasterio.features import shapes

from overnight_dataset import encode, sha
from overnight_drafts import draw, to_rle
from overnight_native_crops import polygon


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("review",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    cfg=json.loads(args.review.read_text())
    for sub in ("images","labels","views"):
        (args.out/sub).mkdir(parents=True,exist_ok=True)
    accepted=[]
    rejected={}
    for kind in ("cam","sign"):
        folder=args.root/f"vn-{kind}-curated"
        src=json.loads((args.root/f"vn-{kind}"/"index.json").read_text())
        rows={r["key"]:r for r in json.loads((folder/"manifest.json").read_text())}
        for key,relative in cfg[kind].items():
            r=rows[key]
            body=(folder/"images"/(key+".jpg")).read_bytes()
            labels=(folder/"labels"/(key+".txt")).read_bytes()
            assert sha(body)==r["image_sha256"] and sha(labels)==r["label_sha256"]
            frame=Image.open(BytesIO(body)).convert("RGB")
            size=np.asarray(frame.size)
            l,t,right,bottom=map(int,np.rint(np.asarray(relative or [0,0,1,1])*np.tile(size,2)))
            crop=frame.crop((l,t,right,bottom))
            w,h=crop.size
            lines=[]; objects=[]
            for line,original in zip(labels.decode().splitlines(),r["objects"],strict=True):
                v=line.split()
                cls=int(v[0])
                if kind=="cam" and r["group"].startswith("day"):
                    cls={0:5,5:0}.get(cls,cls)
                pt=np.asarray(v[1:],float).reshape(-1,2)*size
                mask=mu.decode(to_rle(pt,*size))[t:bottom,l:right]
                if not mask.any():continue
                if relative:
                    coords=polygon(mask)
                    if coords is None:
                        if mask.sum()*(640/max(w,h))**2<4:continue
                        rejected[key]=f"empty mask {original['id']}"
                        break
                else:
                    coords=pt
                fixed=encode(cls,coords,w,h)
                actual=np.asarray(fixed.split()[1:],float).reshape(-1,2)*[w,h]
                iou=float(mu.iou([to_rle(actual,w,h)],[mu.encode(np.asfortranarray(mask))],[0])[0,0])
                if iou<.94 and relative:
                    parts=[g["coordinates"] for g,_ in shapes(mask.astype("uint8"),mask=mask.astype(bool))]
                    if len(parts)==1 and len(parts[0])==1:
                        actual=np.asarray(parts[0][0][:-1])
                        fixed=encode(cls,actual,w,h)
                        rounded=np.asarray(fixed.split()[1:],float).reshape(-1,2)*[w,h]
                        iou=float(mu.iou([to_rle(rounded,w,h)],[mu.encode(np.asfortranarray(mask))],[0])[0,0])
                if iou<.94:
                    if mask.sum()*(640/max(w,h))**2<8:continue
                    rejected[key]=f"clipping IoU {iou:.4f} for {original['id']}"
                    break
                lines.append(fixed)
                objects.append(dict(**{k:v for k,v in original.items() if k not in ("cls","area640")},cls=cls,source_draft_class=int(v[0]),
                                    area640=round(mask.sum()*(640/max(w,h))**2,2),clipping_iou=round(iou,5),poly=actual.tolist()))
            if key in rejected:continue
            assert len(lines)==len(set(lines)) and lines
            if relative:
                buf=BytesIO();crop.save(buf,format="JPEG",quality=95,subsampling=0);body=buf.getvalue()
            label=("\n".join(lines)+"\n").encode()
            (args.out/"images"/(key+".jpg")).write_bytes(body)
            (args.out/"labels"/(key+".txt")).write_bytes(label)
            draw(crop,objects).save(args.out/"views"/(key+".jpg"),quality=93)
            parent=Path("vn-"+kind,"images",r["source_key"]+".jpg")
            accepted.append(dict(key=key,source=src["dataset"],version=src["version"],url=src["url"],license=src["source_license"],
                                 source_image=r["source"],source_sha256=r["source_sha256"],group=r["group"],parent=parent.as_posix(),
                                 parent_sha256=r["prepared_parent_sha256"],roi=[r["roi"][0]+l,r["roi"][1]+t,r["roi"][0]+right,r["roi"][1]+bottom],
                                 image_sha256=sha(body),label_sha256=sha(label),size=[w,h],
                                 qc="Reviewed SAM prompt/crop contact sheets with per-frame corrections, clipped polygon mask checked against raster.",
                                 objects=[{k:v for k,v in o.items() if k!="poly"} for o in objects]))
    (args.out/"manifest.json").write_text(json.dumps(accepted,indent=1))
    (args.out/"rejected.json").write_text(json.dumps(rejected,indent=1))
    print(len(accepted),Counter(o["cls"] for r in accepted for o in r["objects"]),flush=True)


if __name__=="__main__":
    main()
