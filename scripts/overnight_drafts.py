"""Build indexed full-frame drafts and overlays. Review decisions are stored separately."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
from shapely.geometry import LineString, Polygon, box
from shapely.ops import nearest_points, unary_union

NAMES = ["Car","Bus","Truck","Other","Moto","Moto+r","Bike","Bike+r","Ped","Sign","Light"]
COLORS = ["#438bff","#ffa000","#ff4747","#c080ff","#25e6c0","#ff539b","#a6ea38","#ff7216","#13e2fc","#ffd900","#48d16b"]
SOURCE = {"car":0,"bus":1,"truck":2,"rickshaw":3,"cng":3,"easybike":3,"leguna":-1,"van":-1,
          "wheelbarrow":-1,"bike":4,"bicycle":6,"traffic_sign":9}
COCO = {"car":0,"bus":1,"truck":2,"motorcycle":4,"bicycle":6,"person":8,"stop sign":9,"traffic light":10}


def pieces(geom):
    if geom.geom_type=="Polygon": return [geom] if geom.area>.01 else []
    if hasattr(geom,"geoms"):return [p for g in geom.geoms for p in pieces(g)]
    return []


def outline(shape, w, h):
    parts = pieces(shape.intersection(box(0,0,w,h)))
    if not parts:return None
    parts.sort(key=lambda p:p.area, reverse=True)
    shape=parts[0]
    for piece in parts[1:]:
        if piece.area < max(0.8, .001*shape.area):continue
        a,b=nearest_points(shape,piece)
        shape=unary_union([shape,piece, LineString([a,b]).buffer(.6)])
    polygons=pieces(shape.intersection(box(0,0,w,h)))
    if not polygons:return None
    shape=max(polygons,key=lambda x:x.area).simplify(.45,preserve_topology=True)
    if shape.is_empty or shape.area<=1:return None
    # A narrow slit keeps a large internal hole out of a single-ring YOLO polygon.
    for ring in list(shape.interiors):
        if Polygon(ring).area<max(3,.003*shape.area):continue
        a,b=nearest_points(ring,shape.exterior)
        parts=pieces(shape.difference(LineString([a,b]).buffer(.35,cap_style=1)))
        if parts:shape=max(parts,key=lambda s:s.area)
    pts=np.asarray(shape.exterior.coords[:-1])
    if len(pts)<3:return None
    return np.clip(pts,[0,0],[w,h])


def mask_outline(mask):
    h,w=mask.shape
    contours,hierarchy=cv2.findContours(np.asarray(mask,np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    parts=[]
    for i,c in enumerate(contours):
        if hierarchy[0,i,3]!=-1 or len(c)<3:continue
        holes=[]
        hole=hierarchy[0,i,2]
        while hole!=-1:
            if len(contours[hole])>=3:holes.append(contours[hole].reshape(-1,2)+.5)
            hole=hierarchy[0,hole,0]
        ring=Polygon(c.reshape(-1,2)+.5,holes).buffer(0)
        parts.extend(pieces(ring))
    return outline(unary_union(parts),w,h) if parts else None


def to_rle(poly,w,h):
    return mu.frPyObjects([np.asarray(poly).ravel().tolist()],h,w)[0]


def build_record(key,cls,pts,via,w,h):
    if pts is None:return None
    pts=np.round(pts,2)
    if len(pts)<3 or not Polygon(pts).is_valid:return None
    rle=to_rle(pts,w,h)
    area=float(mu.area(rle))
    if area<1:return None
    return dict(id=key,cls=cls,poly=pts.tolist(),via=via,area640=round(area*(640/max(w,h))**2,1))


def road_draft(name, im, prompts, teacher, masks):
    w,h=im.size
    objects=[]
    for a in prompts:
        mask=np.array(Image.open(BytesIO(masks.read(a["mask"]))).convert("1"),dtype=bool)
        pts=mask_outline(mask)
        obj=build_record("s"+str(a["index"]),SOURCE.get(a["source_label"],-1),pts,a["source_label"],w,h)
        if obj:
            target=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
            obj["conversion_iou"]=round(float(mu.iou([to_rle(pts,w,h)],[target],[0])[0,0]),3)
            objects.append(obj)
    for i,a in enumerate(teacher):
        if a["score"]<.26 or not a["polygon"]:continue
        raw=Polygon(a["polygon"]).buffer(0)
        pts=outline(raw,w,h)
        cls=COCO[a["label"]]
        obj=build_record("t"+str(i),cls,pts,"teacher "+a["label"],w,h)
        if not obj or obj["area640"]<45:continue
        a_mask=to_rle(pts,w,h)
        others=[o for o in objects if cls!=8 and o["cls"] not in (8,9,10) or cls==o["cls"]]
        if others:
            rles=[to_rle(o["poly"],w,h) for o in others]
            similarities=mu.iou([a_mask],rles,[0]*len(rles))[0]
            if max(similarities)>.28:continue
        obj["confidence"]=a["score"]
        objects.append(obj)
    return objects


def coco_draft(im,rows):
    w,h=im.size
    result=[]
    for j,a in enumerate(rows):
        parts=[Polygon(np.asarray(p).reshape(-1,2)).buffer(0) for p in a["segmentation"] if len(p)>=6]
        if not parts:continue
        pts=outline(unary_union(parts),w,h)
        obj=build_record("s"+str(j),COCO[a["label"]],pts,"COCO "+str(a["id"]),w,h)
        if obj:
            native=mu.merge(mu.frPyObjects(a["segmentation"],h,w))
            obj["conversion_iou"]=round(float(mu.iou([to_rle(pts,w,h)],[native],[0])[0,0]),3)
            result.append(obj)
    return result


def rider_drafts(objects,w,h):
    persons=[o for o in objects if o["cls"]==8]
    bikes=[o for o in objects if o["cls"] in (4,6)]
    candidates=[]
    for bike in bikes:
        vehicle=Polygon(bike["poly"])
        x1,y1,x2,y2=vehicle.bounds
        bw,bh=max(x2-x1,1),max(y2-y1,1)
        for rider in persons:
            body=Polygon(rider["poly"])
            a,b,c,d=body.bounds
            cx=(a+c)/2
            center=(cx-x1)/bw
            relative=((b+d)/2-y1)/bh
            seat=(d-y1)/bh
            if not -.10<=center<=1.10 or not -.95<=relative<=.58 or not -.05<=seat<=1.17:continue
            near=vehicle.distance(body)/max(bh,bw)
            if near>.11:continue
            candidates.append((1-abs(center-.52)-.4*abs(seat-.7)-2*near,bike,rider))
    used=set()
    for _,bike,rider in sorted(candidates,key=lambda r:-r[0]):
        if bike["id"] in used or rider["id"] in used:continue
        used.update([bike["id"],rider["id"]])
        pts=outline(unary_union([Polygon(bike["poly"]),Polygon(rider["poly"])]),w,h)
        merged=build_record(bike["id"]+"+"+rider["id"],bike["cls"]+1,pts,"draft rider association",w,h)
        if merged:
            merged["components"]=[bike,rider]
            objects.remove(bike);objects.remove(rider);objects.append(merged)
    return objects


def draw(im,objects,scale=1):
    view=im.convert("RGB").copy()
    view.thumbnail((int(850*scale),int(575*scale)))
    w,h=im.size
    sx,sy=view.width/w,view.height/h
    layer=Image.new("RGBA",view.size)
    canvas=ImageDraw.Draw(layer)
    labels=[]
    try:font=ImageFont.truetype("arial.ttf",max(10,int(12*scale)))
    except OSError:font=ImageFont.load_default()
    for o in objects:
        pts=np.asarray(o["poly"])*[sx,sy]
        xy=[tuple(v) for v in pts]
        color=COLORS[o["cls"]] if o["cls"]>=0 else "#ffffff"
        canvas.polygon(xy,fill=color+"23")
        canvas.line(xy+[xy[0]],fill=color,width=2)
        if o["area640"]>=55:
            cx,cy=Polygon(pts).representative_point().coords[0]
            labels.append((cx,cy,o["id"]+"/"+str(o["cls"]),color))
    view=Image.alpha_composite(view.convert("RGBA"),layer).convert("RGB")
    canvas=ImageDraw.Draw(view)
    for x,y,label,color in labels:canvas.text((max(0,x-8),max(0,y-6)),label,font=font,fill=color,stroke_width=1,stroke_fill="black")
    return view


def main():
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=["bd","vn","coco"])
    p.add_argument("images",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--masks",type=Path)
    p.add_argument("--teacher",type=Path)
    a=p.parse_args()
    for sub in ["images","objects","views"]:
        (a.out/sub).mkdir(exist_ok=True,parents=True)
    z=ZipFile(a.images)
    meta=json.loads(z.read("manifest.json"))
    rows=meta if isinstance(meta,list) else meta["records"]
    masks=ZipFile(a.masks) if a.masks else None
    sources=json.loads(masks.read("source-objects.json")) if masks else {}
    teacher=json.loads(a.teacher.read_text())["images"] if a.teacher else {}
    index=[]
    for i,row in enumerate(rows):
        name=row["file"];key=Path(name).stem
        if a.kind=="bd":key=f"bd{int(key.rsplit('_',1)[1]):03d}"
        elif a.kind=="vn":key=f"vn{i:03d}"
        else:key=f"co{i:03d}"
        body=z.read("images/"+name)
        im=Image.open(BytesIO(body)).convert("RGB")
        objects=coco_draft(im,row["objects"]) if a.kind=="coco" else road_draft(name,im,sources[name],teacher[name],masks)
        objects=rider_drafts(objects,*im.size)
        (a.out/"images"/(key+".jpg")).write_bytes(body)
        (a.out/"objects"/(key+".json")).write_text(json.dumps(objects))
        draw(im,objects).save(a.out/"views"/(key+".jpg"),quality=91)
        index.append(dict(key=key,name=name,w=im.width,h=im.height,instances=len(objects),
                          meta={k:v for k,v in row.items() if k not in ("objects","boxes")}))
    (a.out/"index.json").write_text(json.dumps(index,indent=1))
    print(a.kind,len(index),sum(r["instances"] for r in index),flush=True)


if __name__=="__main__":
    main()
