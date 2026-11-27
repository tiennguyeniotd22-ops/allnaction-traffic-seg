"""Generate SAM box prompts for road signs and publisher-labeled Vietnam vehicles."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image


def sign_boxes(im):
    w, h = im.size
    fit = min(1, 900 / max(w, h))
    rgb = cv2.resize(np.asarray(im), None, fx=fit, fy=fit, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    bands = [((hue <= 12) | (hue >= 164)) & (sat >= 70) & (val > 39),
             (hue >= 94) & (hue <= 132) & (sat > 57) & (val > 29),
             (hue >= 15) & (hue <= 38) & (sat > 80) & (val > 55)]
    candidates = []
    for band in bands:
        close = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if min(bw,bh) < 11 or not .19 < bw/bh < 5 or bw*bh > rgb.shape[0]*rgb.shape[1]*.46:
                continue
            hull = cv2.convexHull(contour)
            fill = cv2.contourArea(hull)/(bw*bh)
            solid = cv2.contourArea(contour)/max(1, cv2.contourArea(hull))
            if fill < .43 or solid < .34:
                continue
            score = (bw*bh)**.4 * fill**2 * solid**2 * (1.16-.26*y/rgb.shape[0])
            margin = max(1.5, min(bw,bh)*.025)
            bbox = np.array([x-margin,y-margin,x+bw+margin,y+bh+margin])/fit
            candidates.append((score, np.clip(bbox,[0,0,0,0],[w,h,w,h])))
    kept = []
    for score, b in sorted(candidates, key=lambda k:-k[0]):
        area = np.prod(b[2:]-b[:2])
        if any(np.prod(np.maximum(0,np.minimum(b[2:],q[2:])-np.maximum(b[:2],q[:2]))) /
               (area + np.prod(q[2:]-q[:2]) - np.prod(np.maximum(0,np.minimum(b[2:],q[2:])-np.maximum(b[:2],q[:2])))) > .55 for q in kept):
            continue
        kept.append(b)
        if len(kept) == 7:
            break
    return [("sign_proposal",b) for b in kept]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("cam","sign"))
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    stage=args.root/("vn-"+args.kind)
    rows=json.loads((stage/"index.json").read_text())["records"]
    records=[]
    with ZipFile(args.out, "x", ZIP_DEFLATED) as result:
        for i,row in enumerate(rows):
            name=f"{args.kind[0]}{i:03}.jpg"
            body=(stage/"images"/(row["key"]+".jpg")).read_bytes()
            im=Image.open(BytesIO(body)).convert("RGB")
            w,h=im.size
            boxes=[]
            if args.kind=="cam":
                for c,x,y,bw,bh in row["boxes"]:
                    if bw*bh*w*h*(640/max(w,h))**2 < 32:
                        continue
                    b=np.clip(np.array([x-bw/2,y-bh/2,x+bw/2,y+bh/2])*[w,h,w,h],[0,0,0,0],[w,h,w,h])
                    if min(b[2:]-b[:2])>2:
                        boxes.append((["car","moto_or_rider","bus","truck"][int(c)],b))
            else:
                boxes=sign_boxes(im)
            if not boxes:
                continue
            xml=ET.Element("annotation")
            for label,b in boxes:
                obj=ET.SubElement(xml,"object")
                ET.SubElement(obj,"name").text=label
                coords=ET.SubElement(obj,"bndbox")
                for key,v in zip(("xmin","ymin","xmax","ymax"),b):
                    ET.SubElement(coords,key).text=f"{v:.3f}"
            result.writestr("images/"+name,body)
            result.writestr("source_xml/"+name[:-4]+".xml",ET.tostring(xml))
            records.append(dict(file=name,original_size=[w,h],size=[w,h],source_key=row["key"]))
        result.writestr("manifest.json",json.dumps({"records":records}))
    print(args.kind,len(records),flush=True)


if __name__=="__main__":
    main()
