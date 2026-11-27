"""Apply recorded image-level decisions and build a versioned, val-preserving ZIP."""
import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from pycocotools import mask as mu

from overnight_drafts import draw, outline, to_rle


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_member(archive, name, data):
    member = ZipInfo(name, date_time=(2026, 11, 27, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o100644 << 16
    archive.writestr(member, data)


def encode(cls, points, w, h):
    norm = np.round(np.asarray(points)/[w,h], 7)
    if not np.isfinite(norm).all() or np.min(norm)<0 or np.max(norm)>1:
        raise ValueError("Coordinates outside the image")
    if not Polygon(norm).is_valid or len(set(map(tuple,norm)))<3:
        raise ValueError("Invalid serialized ring")
    return str(cls)+" "+" ".join(f"{v:.7f}" for v in norm.ravel())


def curate(key, folder, row, decisions, roi, class3_only):
    body=(folder/"images"/(key+".jpg")).read_bytes()
    source=Image.open(BytesIO(body)).convert("RGB")
    w,h=source.size
    assert (w,h)==(row["w"],row["h"])
    objects={o["id"]:o for o in read(folder/"objects"/(key+".json"))}
    for name in decisions["drop"].get(key,[]):
        objects.pop(name)
    for name,cls in decisions["reclass"].get(key,{}).items():
        objects[name]["cls"]=cls
    for name in list(objects):
        if key+"/"+name in decisions["split"]:
            for o in objects.pop(name)["components"]:
                objects[o["id"]]=o
    for a,b,cls in decisions["merge"].get(key,[]):
        merged=unary_union([Polygon(objects.pop(n)["poly"]) for n in (a,b)])
        objects[a+"+"+b]={"id":a+"+"+b,"cls":cls,"poly":outline(merged,w,h),"via":"visually confirmed union"}
    for name,extra in decisions.get("add",{}).get(key,{}).items():
        objects[name]={"id":name,"cls":extra["cls"],"poly":extra["poly"],"via":"manually traced visible sign face"}
    xy=np.round(np.asarray(roi)*[w,h,w,h]).astype(int)
    crop=source.crop(tuple(xy))
    cw,ch=crop.size
    geom=box(*xy)
    labels=[]
    result=[]
    for o in objects.values():
        cls=o["cls"]
        clipped=Polygon(o["poly"]).intersection(geom)
        if clipped.area<=1 or class3_only and cls!=3:continue
        if cls<0:
            if clipped.area*(640/max(cw,ch))**2>40:
                raise ValueError("unresolved source class "+o["id"])
            continue
        if key.startswith("co") and min([s.get("conversion_iou",1) for s in o.get("components",[o])])<.92:
            raise ValueError("source-mask IoU under .92")
        pts=outline(clipped,w,h) if not geom.covers(Polygon(o["poly"])) else np.asarray(o["poly"])
        if pts is None:continue
        pts=np.clip(pts-xy[:2],[0,0],[cw,ch])
        area=int(mu.area(to_rle(pts,cw,ch)))
        if area==0:continue
        labels.append(encode(cls,pts,cw,ch))
        result.append(dict(id=o["id"],cls=cls,poly=pts.tolist(),area640=round(area*(640/max(cw,ch))**2,3),
                           via=o["via"],conversion_iou=o.get("conversion_iou")))
    if not result:raise ValueError("no usable masks")
    if list(roi)!=[0,0,1,1]:
        buf=BytesIO();crop.save(buf,format="JPEG",quality=95,subsampling=0);body=buf.getvalue()
    label=("\n".join(labels)+"\n").encode()
    if len(set(labels))!=len(labels):raise ValueError("duplicate polygon")
    return body,label,crop,result


def repair_base(data,imsize,rows):
    lines=data.decode().splitlines()
    log=[]
    for idx in rows:
        v=lines[idx-1].split()
        pts=np.asarray(v[1:],float).reshape(-1,2)*imsize
        old=to_rle(pts,*imsize)
        shape=Polygon(pts)
        if shape.is_valid:continue
        fixed=outline(shape.buffer(0),*imsize)
        if fixed is None:continue
        fit=float(mu.iou([old],[to_rle(fixed,*imsize)],[0])[0,0])
        if fit<.985:continue
        lines[idx-1]=encode(int(v[0]),fixed,*imsize)
        log.append({"row":idx,"repair":"remove self-crossing","iou":fit})
    return ("\n".join(lines)+"\n").encode(),log


def main():
    p=argparse.ArgumentParser()
    for name in ("base","root","decisions","manifest","queue","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    cfg=read(args.decisions)
    holdout=set(cfg["exclude_from_train"])
    assert all(int(k[2:]) not in holdout for k in cfg["bd_train_windows"])
    assert all(int(k[2:]) in holdout for k in cfg["holdout_windows"])
    out=args.out
    out.mkdir(exist_ok=True,parents=True)
    result=[];reject={};changes=[]
    val={r["file"]:r for r in read(args.manifest)}
    repair={}
    with args.queue.open() as f:
        for r in csv.DictReader(f):
            if r["issue"]=="proper_self_crossing":repair.setdefault(r["image"],[]).append(int(r["details"].split()[-1]))
    remove={r["file"] for r in read(args.root/"team/index.json") if r["key"]=="team191" and r["split"]=="train"}
    pairs=read(args.root/"scene-matches.json")
    train_sha=set()
    with ZipFile(args.base) as original, ZipFile(out/"dataset.partial","w",ZIP_DEFLATED) as dest, ZipFile(out/"holdout.partial","w",ZIP_DEFLATED) as held:
        assert {Path(n).name for n in original.namelist() if n.startswith("images/val/")} == set(val)
        assert {Path(n).stem for n in original.namelist() if n.startswith("labels/val/")} == {Path(n).stem for n in val}
        for name in original.namelist():
            data=original.read(name)
            f=Path(name)
            if name.startswith("images/"):
                train_sha.add(sha(data))
                if "/val/" in name:assert sha(data)==val[f.name]["image_sha256"]
            elif name.startswith("labels/") and f.suffix==".txt":
                jpg=f.stem+".jpg"
                if "/val/" in name:assert sha(data)==val[jpg]["labels_sha256"]
                elif jpg in repair or jpg in remove:
                    size=Image.open(BytesIO(original.read("images/train/"+jpg))).size
                    data,log=repair_base(data,size,repair.get(jpg,[]))
                    if jpg in remove:
                        lines=data.decode().splitlines(); assert lines[3].split()[0]=="8"
                        del lines[3];data=("\n".join(lines)+"\n").encode();log.append(dict(row=4,repair="remove advertising poster pedestrian"))
                    changes.extend(dict(image=jpg,**x) for x in log)
            write_member(dest,name,data)
        visited=set()
        for source,window,role in [("coco",{f"co{i:03}":[0,0,1,1] for i in cfg["coco_train"]},"train"),
                                 ("bd",cfg["bd_train_windows"],"train"),("vn",cfg.get("vn_train_windows",{}),"train"),
                                 ("bd",cfg["holdout_windows"],"holdout")]:
            folder=args.root/source
            index={r["key"]:r for r in read(folder/"index.json")}
            for key,roi in window.items():
                row=index[key]
                if role=="train":
                    related={p["b"] if p["a"]==key else p["a"] for p in pairs if key in (p["a"],p["b"]) and p["inliers"]>=15}
                    if any(v in visited or v.startswith("images/val/") or v in cfg["holdout_windows"] for v in related):
                        reject[key]="similar scene already used or reserved";continue
                try:
                    body,label,crop,objects=curate(key,folder,row,cfg,roi,role=="holdout")
                except ValueError as e:
                    reject[key]=str(e);continue
                if sha(body) in train_sha:raise ValueError("duplicate image "+key)
                train_sha.add(sha(body));visited.add(key)
                archive,split=(dest,"train") if role=="train" else (held,"val")
                stem="zz_new_"+key
                write_member(archive,f"images/{split}/{stem}.jpg",body)
                write_member(archive,f"labels/{split}/{stem}.txt",label)
                viewdir=out/("views-"+role);viewdir.mkdir(exist_ok=True)
                draw(crop,objects).save(viewdir/(key+".jpg"),quality=92)
                result.append(dict(key=key,role=role,stem=stem,source=source,roi=roi,source_meta=row["meta"],image_sha256=sha(body),
                                   label_sha256=sha(label),objects=[{k:v for k,v in o.items() if k!="poly"} for o in objects]))
    for name in ("dataset","holdout"):
        temp=out/(name+".partial")
        with ZipFile(temp) as z:assert z.testzip() is None
        temp.replace(out/(name+".zip"))
    counts={role:dict(Counter(o["cls"] for r in result if r["role"]==role for o in r["objects"])) for role in ("train","holdout")}
    report=dict(base_sha256=sha(args.base.read_bytes()),validation_images_verified=len(val),records=result,excluded=reject,
                edits=changes,added_instances=counts,dataset_sha256=sha((out/"dataset.zip").read_bytes()),holdout_sha256=sha((out/"holdout.zip").read_bytes()))
    (out/"manifest.json").write_text(json.dumps(report,indent=1),encoding="utf8")
    print(Counter((r["role"],r["source"]) for r in result),counts,"edits",len(changes),"excluded",reject,flush=True)


if __name__=="__main__":
    main()
