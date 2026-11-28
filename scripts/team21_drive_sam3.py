"""Inspect and prepare a team-provided YOLO segmentation ZIP, retaining the frozen validation split."""
import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as masks
from shapely.geometry import Polygon
import yaml

from overnight_dataset import encode, sha, write_member
from overnight_drafts import COLORS, outline, to_rle
from overnight_extend import check_labels
TARGETS = ("Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light")


def save(path,data):
    path.write_bytes((json.dumps(data,indent=1,ensure_ascii=False)+"\n").encode())


def inspect(folder):
    out=folder/"curated"
    for sub in ("images","labels","views","sheets"):
        (out/sub).mkdir(exist_ok=True,parents=True)
    source=folder/"source.zip"
    assert sha(source.read_bytes())=="dea796efad0f105318f6a97634e9d981c186e0f6e6d8fe36ab4c9e30fdc217a0"
    records=[]
    skipped=[]
    with ZipFile(source) as z:
        config=yaml.safe_load(z.read("data.yaml"))
        assert tuple(config["names"][i] for i in range(11))==TARGETS
        names=sorted(n for n in z.namelist() if n.startswith("images/train/") and n.endswith(".png"))
        assert len(names)==26 and len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        for name in names:
            key=Path(name).stem
            assert key.startswith("vn_light_") and key[9:].isdigit()
            body=z.read(name)
            im=Image.open(BytesIO(body)).convert("RGB")
            w,h=im.size
            lines=z.read("labels/train/"+key+".txt").decode().splitlines()
            items=[]
            labels=[]
            for index,line in enumerate(lines):
                fields=line.split()
                c=int(fields[0])
                assert 0<=c<11
                xy=np.array(fields[1:],float).reshape(-1,2)
                if not np.isfinite(xy).all() or np.any((xy<0)|(xy>1)): raise ValueError((name,index))
                xy *= [w,h]
                poly=Polygon(xy)
                reason=None
                if np.ptp(xy,axis=0).min()<.0125*max(w,h): reason="below BTC short-side guideline"
                elif not poly.is_valid:
                    candidate=outline(poly.buffer(0),w,h)
                    if candidate is None or masks.iou([to_rle(xy,w,h)],[to_rle(candidate,w,h)],[0])[0,0]<.97:
                        reason="ambiguous self-intersection"
                    else: xy=candidate
                if reason is None:
                    try:
                        serialized=encode(c,xy,w,h)
                        check_labels((serialized+"\n").encode(),[dict(cls=c)])
                    except ValueError:
                        reason="invalid serialized polygon"
                if reason is not None:
                    skipped.append(dict(key=key,row=index,cls=c,reason=reason))
                    continue
                items.append(dict(cls=c,row=index))
                labels.append(serialized)
            assert len(items)==len(labels)
            buf=BytesIO()
            im.save(buf,"JPEG",quality=97,subsampling=0)
            jpg=buf.getvalue()
            (out/"images"/(key+".jpg")).write_bytes(jpg)
            parent=folder/"parents"/(key+".png")
            parent.parent.mkdir(exist_ok=True)
            parent.write_bytes(body)
            material=("\n".join(labels)+"\n").encode()
            (out/"labels"/(key+".txt")).write_bytes(material)
            view=im.copy()
            draw=ImageDraw.Draw(view)
            for line,item in zip(labels,items,strict=True):
                xy=np.array(line.split()[1:],float).reshape(-1,2)*[w,h]
                pts=list(map(tuple,xy))
                c=item["cls"]
                draw.line(pts+pts[:1],fill=COLORS[c],width=2)
                if c in (4,5,6,7): draw.text(tuple(xy.mean(0)),str(c)+"/"+str(item["row"]),fill="white",stroke_width=1,stroke_fill="black")
            view.thumbnail((1000,650))
            view.save(out/"views"/(key+".jpg"),quality=93)
            records.append(dict(key=key,objects=items,size=[w,h],image_sha256=sha(jpg),label_sha256=sha(material),
              parent=parent.relative_to(folder.parent).as_posix(),parent_sha256=sha(body),source_image=name,
              source="User-shared SAM3 prelabels",license="Team-supplied; upstream rights unspecified"))
    for start in range(0,len(records),6):
        sheet=Image.new("RGB",(1200,3*238),"white")
        draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(records[start:start+6]):
            with Image.open(out/"views"/(row["key"]+".jpg")) as im:
                im.thumbnail((595,211))
                x,y=j%2*600,j//2*238
                sheet.paste(im,(x+(600-im.width)//2,y))
                draw.text((x+4,y+213),row["key"]+" "+str(dict(Counter(o["cls"] for o in row["objects"]))),fill="black")
        sheet.save(out/"sheets"/f"{start:02}.jpg",quality=94)
    save(out/"manifest.json",records)
    save(folder/"geometry-audit.json",skipped)
    print(len(records),sum(len(r["objects"]) for r in records),"accepted polygons", Counter(s["reason"] for s in skipped))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    inspect(p.parse_args().folder)
