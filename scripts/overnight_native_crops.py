"""Convert visible panoptic instances to reviewed candidate YOLO polygons."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from itertools import repeat

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu
from rasterio.features import shapes
from shapely import affinity
from shapely.geometry import shape
from shapely.ops import unary_union

from overnight_dataset import encode
from overnight_drafts import COLORS, outline, to_rle


CLASS = {26:0,28:1,27:2,32:4,33:6,24:8,20:9,19:10,25:-2,31:-1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def polygon(mask):
    y,x=np.nonzero(mask)
    if not len(x):return None
    left,top=x.min(),y.min()
    region=mask[top:y.max()+1,left:x.max()+1].astype(np.uint8)
    parts=[shape(g) for g,_ in shapes(region,mask=region.astype(bool))]
    geometry=affinity.translate(unary_union([p if p.is_valid else p.buffer(0) for p in parts]),int(left),int(top))
    return outline(geometry,mask.shape[1],mask.shape[0])


def semantic_components(src,items):
    compact=np.zeros(src.shape,np.int32)
    info={}
    for code,cat,crowd in items:
        if cat not in CLASS:continue
        where=src==code
        if not where.any():continue
        if cat in {19,20}:
            n,split,stats,_=cv2.connectedComponentsWithStats(where.astype(np.uint8),4)
            for i in range(1,n):
                x,y,w,h,area=map(int,stats[i])
                if area<3:continue
                idx=len(info)+1
                compact[y:y+h,x:x+w][split[y:y+h,x:x+w]==i]=idx
                info[idx]=(CLASS[cat],[x,y,x+w,y+h],code)
        else:
            ys,xs=np.nonzero(where)
            idx=len(info)+1
            info[idx]=(-1 if crowd else CLASS[cat],[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],code)
            compact[where]=idx
    return compact,info


def bdd_components(mask):
    # BDD semantic train IDs. Touching objects with the same ID require visual review.
    mapping = {6:10, 7:9, 11:8, 12:-2, 13:0, 14:2, 15:1, 16:-1, 17:4, 18:6}
    compact = np.zeros(mask.shape, np.int32)
    objects = {}
    for code, cls in mapping.items():
        count, regions, stats, _ = cv2.connectedComponentsWithStats((mask == code).astype(np.uint8), connectivity=8)
        for component in range(1, count):
            x, y, w, h, area = map(int, stats[component])
            if area < 3:
                continue
            index = len(objects) + 1
            compact[y:y+h, x:x+w][regions[y:y+h, x:x+w] == component] = index
            objects[index] = cls, [x,y,x+w,y+h], f"{code}-{component}"
    return compact, objects


def bdd_windows(w, h, objects, target):
    for cls, (x,y,xx,yy), _ in objects.values():
        if cls != target or min(xx-x,yy-y)<7:
            continue
        for context in (2.2, 3.8):
            cw = min(w, max(260, round(max(xx-x, yy-y)*context)))
            ch = min(h, max(195, round(cw*.70), yy-y+32))
            left = max(0, min(w-cw, (x+xx-cw)//2))
            top = max(0, min(h-ch, (y+yy-ch)//2))
            yield left, top, left+cw, top+ch


def associate(pixels,objects):
    bikes={i:b for i,(cls,b,_) in objects.items() if cls in (4,6)}
    assigned={}
    for i,(cls,(x,y,xx,yy),_) in objects.items():
        if cls not in (-2, -3, -4):continue
        rider_class = cls
        scores=[]
        for k,(l,t,r,b) in bikes.items():
            if rider_class == -3 and objects[k][0] != 6 or rider_class == -4 and objects[k][0] != 4:
                continue
            w,h=r-l,b-t
            center=((x+xx)/2-l)/w
            foot=(yy-t)/h
            inter=max(0,min(xx,r)-max(x,l))/max(1,min(xx-x,w))
            if not -.18<=center<=1.18 or not -.27<=foot<=1.3 or y>=t+.3*h or inter<.12:continue
            gap=max(0,t-yy)/h
            scores.append((inter-.52*abs(center-.5)-.32*abs(foot-.65)-1.5*gap,k))
        scores.sort(reverse=True)
        if scores and scores[0][0]>.15 and (len(scores)==1 or scores[0][0]-scores[1][0]>.14):
            assigned.setdefault(scores[0][1],[]).append(i)
        else:
            for _,k in scores:
                cls,bb,code=objects[k];objects[k]=(-1,bb,code)
    for k,riders in assigned.items():
        cls,bb,code=objects[k]
        if cls not in (4,6):continue
        ids=[code]
        for i in riders:
            _,b,c=objects.pop(i);ids.append(c)
            pixels[pixels==i]=k
            bb=[min(b[0],bb[0]),min(b[1],bb[1]),max(b[2],bb[2]),max(b[3],bb[3])]
        objects[k]=(cls+1,bb,ids)


def vistas_instances(mask, row, categories):
    # The panoptic JSON has its own category IDs. Source config_v2.0.json is
    # indexed differently. Read the semantic names from the paired panoptic JSON.
    rgb = np.asarray(mask.convert("RGB"), dtype=np.int32)
    native = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    targets = {"human--person--individual": 8, "human--rider--bicyclist": -3,
               "human--rider--motorcyclist": -4, "human--rider--other-rider": -1,
               "object--vehicle--bicycle": 6, "object--vehicle--bus": 1,
               "object--vehicle--car": 0, "object--vehicle--motorcycle": 4,
               "object--vehicle--truck": 2, "object--vehicle--trailer": 3,
               "object--vehicle--other-vehicle": -1, "object--vehicle--caravan": -1,
               "object--vehicle--wheeled-slow": -1, "void--ego-vehicle": -1}
    counts = np.bincount(native.ravel())
    lut = np.zeros(len(counts), dtype=np.int16)
    objects = {}
    for segment in row["segments"]:
        uid = segment["id"]
        assert counts[uid] == segment["area"]
        name = categories[str(segment["category_id"])]
        if name.startswith("object--traffic-light--"):
            cls = 10
        elif name.startswith("object--traffic-sign--"):
            cls = 9
        elif name in targets:
            cls = targets[name]
        else:
            continue
        x, y, w, h = segment["bbox"]
        idx = len(objects) + 1
        objects[idx] = -1 if segment["iscrowd"] else cls, [x,y,x+w,y+h], uid
        lut[uid] = idx
    return lut[native], objects


def candidates(w,h,objects):
    yield 0,0,w,h
    # Prefer broad front-facing scenes. Narrow windows isolate an ambiguous crowd
    # or let a small distant traffic participant retain useful detail at 640.
    for scale in (.70,.46):
        cw=max(1,round(w*scale));ch=min(h,max(round(h*.62),round(cw*.62)))
        for fx in (.15,.5,.85):
            x=max(0,min(w-cw,round(w*fx-cw/2)))
            y=max(0,min(h-ch,round(h*.53-ch/2)))
            yield x,y,x+cw,y+ch
    for cls,(l,t,r,b),_ in objects.values():
        if cls not in (1,2,4,5,6,7,9,10) or min(r-l,b-t)<6:continue
        cw=min(w,max(round(w*.25),(r-l)*3,(b-t)*3))
        ch=min(h,max(round(h*.32),round(cw*.72)))
        x=max(0,min(w-cw,(l+r-cw)//2));y=max(0,min(h-ch,(t+b-ch)//2))
        yield x,y,x+cw,y+ch


def process(row,kind,root,out,categories=None):
    cv2.setNumThreads(1)
    key=row["key"]
    a=(root/"native"/(key+".png")).read_bytes()
    suffix = Path(row["image"]).suffix
    b=(root/"images"/(key+suffix)).read_bytes()
    assert digest(a)==row["annotation_sha256"] and digest(b)==row["image_sha256"]
    source=Image.open(BytesIO(b)).convert("RGB")
    w,h=source.size
    mask=Image.open(BytesIO(a))
    assert mask.size==source.size
    if kind=="bdd-sem":
        pixels, objects = bdd_components(np.asarray(mask))
    elif kind=="vistas":
        pixels, objects = vistas_instances(mask, row, categories)
    elif kind=="kitti":
        grid=np.asarray(mask)
        items=[(int(v),int(v)//256,int(v)%256==0) for v in np.unique(grid)]
    else:
        rgb=np.asarray(mask.convert("RGB"),dtype=np.int32)
        grid=rgb[...,0]+256*rgb[...,1]+65536*rgb[...,2]
        items=[(s["id"],s["category_id"],bool(s["iscrowd"])) for s in row["segments"]]
    if kind not in ("bdd-sem", "vistas"):
        pixels,objects=semantic_components(grid,items)
    associate(pixels,objects)
    options=[]
    if kind=="bdd-sem":
        target = {"17-empty":4, "17-rider":5, "18-empty":6, "18-rider":7, "15":1, "6":10, "7":9}[row["bucket"]]
        windows = bdd_windows(w,h,objects,target)
    else:
        windows = candidates(w,h,objects)
    for roi in dict.fromkeys(windows):
        x,y,xx,yy=roi;cw,ch=xx-x,yy-y
        stride=max(1,round(max(w,h)/900)) if kind=="vistas" else 1
        region=pixels[y:yy:stride,x:xx:stride]
        counts=np.bincount(region.ravel())
        measures=[(int(i),float(counts[i])*(640*stride/max(cw,ch))**2) for i in np.flatnonzero(counts) if i]
        if any(objects[i][0]<0 and area>=24 for i,area in measures):continue
        relevant=[(objects[i][0],area) for i,area in measures if area>=64 and objects[i][0]>=0]
        if not relevant:continue
        rare={c for c,area in relevant if c in (1,2,4,5,6,7,9,10) and area>=110}
        score=2.15*len(rare)+1.8*np.sqrt(cw*ch/(w*h))+min(9,len(relevant))*.18
        if kind=="vistas":
            focus = {"bicycle":6, "bicyclist":7, "motorcycle":4, "motorcyclist":5,
                     "bus":1, "direction-front":9, "general-single":10, "trailer":3}[row["selection"]]
            if not any(c == focus and area >= 140 for c,area in relevant) or len(relevant) > 19:
                continue
            score += 2.5*np.sqrt(cw*ch/(w*h)) - .13*len(relevant)
        if kind=="bdd-sem":
            target_areas = [area for c,area in relevant if c==target]
            if not target_areas or max(target_areas)<250 or len(relevant)>10:
                continue
            score = min(np.log2(max(target_areas)), 13) + min(1.5,cw/500) - .17*len(relevant)
        options.append((score,roi,measures))
    for _,roi,measures in sorted(options,key=lambda e:-e[0])[:8]:
        x,y,xx,yy=roi;cw,ch=xx-x,yy-y
        region=pixels[y:yy,x:xx]
        if kind=="vistas":
            counts=np.bincount(region.ravel())
            measures=[(int(i),float(counts[i])*(640/max(cw,ch))**2) for i in np.flatnonzero(counts) if i]
            if any(objects[i][0]<0 and area>=24 for i,area in measures):
                continue
        rows=[];labels=[];kept=[];tiny=0;failed=False
        for i,area in measures:
            cls,bb,native_id=objects[i]
            if cls<0 or area<1:tiny+=1;continue
            native=region==i
            try:
                pts=polygon(native)
                if pts is None: raise ValueError("empty ring")
                line=encode(cls,pts,cw,ch)
                norm=np.asarray(line.split()[1:],float).reshape(-1,2)
                target=mu.encode(np.asfortranarray(native,dtype=np.uint8))
                fit=float(mu.iou([to_rle(norm*[cw,ch],cw,ch)],[target],[0])[0,0])
                if fit<(.94 if area>=64 else .8):raise ValueError("mask conversion loss")
            except (ValueError,ArithmeticError):
                if area>=18:
                    failed=True;break
                tiny+=1;continue
            labels.append(line)
            rows.append(dict(cls=cls,native_id=native_id,iou=round(fit,5),area640=round(area,2)))
            kept.append((cls,pts))
        if failed or not rows:continue
        image=source.crop(roi);buffer=BytesIO();image.save(buffer,format="JPEG",quality=94,subsampling=0)
        img=buffer.getvalue();lbl=("\n".join(labels)+"\n").encode()
        (out/"images"/(key+".jpg")).write_bytes(img)
        (out/"labels"/(key+".txt")).write_bytes(lbl)
        overlay=image.copy();overlay.thumbnail((615,340));display=ImageDraw.Draw(overlay)
        for cls,pts in kept:
            scaled=[tuple(v) for v in pts*[overlay.width/cw,overlay.height/ch]]
            display.line(scaled+[scaled[0]],fill=COLORS[cls],width=2)
        overlay.save(out/"views"/(key+".jpg"),quality=89)
        return dict(key=key,roi=list(roi),size=[cw,ch],source_sha256=row["image_sha256"],source_mask_sha256=row["annotation_sha256"],
                    image_sha256=digest(img),label_sha256=digest(lbl),source_image=row["image"],source_mask=row["annotation"],
                    drive=row.get("drive"),condition=row.get("condition"),objects=rows,omitted_tiny_or_unresolved_below_cutoff=tiny,
                    **({"selection":row["bucket"]} if kind=="bdd-sem" else {"selection":row["selection"]} if kind=="vistas" else {}))
    return dict(key=key,rejected="ambiguous source regions, small objects or polygon fit")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=("kitti","muses","bdd-sem","vistas"));p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--index",type=Path,help="Use a verified snapshot of complete pairs")
    a=p.parse_args()
    for name in ("images","labels","views"):(a.out/name).mkdir(parents=True,exist_ok=True)
    index=json.loads((a.index or a.root/"index.json").read_text())
    records=index["records"]
    with ProcessPoolExecutor(a.workers) as pool:
        result=list(pool.map(process,records,repeat(a.kind),repeat(a.root),repeat(a.out),repeat(index.get("categories")),chunksize=3))
    (a.out/"manifest.json").write_text(json.dumps(result,indent=1))
    print(len(result),sum("roi" in r for r in result),Counter(o["cls"] for r in result for o in r.get("objects",[])),flush=True)


if __name__=="__main__":
    main()
