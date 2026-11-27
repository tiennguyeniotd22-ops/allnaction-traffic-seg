"""Select held-apart source videos for Vietnamese sign QC and package their box prompts."""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import Image
from stage_poribohon import dhash


def video(name):
    match = re.match(r"(.+?)[-_](?:mov|mp4)_out\d+", name, re.I)
    return match.group(1).lower() if match else None


def main():
    p = argparse.ArgumentParser()
    for arg in ("export","used","baseline","out"):
        p.add_argument(arg,type=Path)
    a = p.parse_args()
    prior = {video(row["original"]) for row in json.loads(a.used.read_text())["records"]}
    fingerprints = []
    with ZipFile(a.baseline) as z:
        for path in z.namelist():
            if path.endswith(".jpg"):
                fingerprints.append(dhash(Image.open(BytesIO(z.read(path)))))
    with ZipFile(a.export) as z, ZipFile(a.out,"x") as result:
        path = next(n for n in z.namelist() if "/train/" in n and n.endswith(".json"))
        prefix = path.rsplit("/",1)[0]+"/"
        src = json.loads(z.read(path))
        anns = defaultdict(list)
        for obj in src["annotations"]:
            anns[obj["image_id"]].append(obj)
        groups = defaultdict(list)
        for im in src["images"]:
            v = video(im["file_name"])
            if not v or v in prior or not anns[im["id"]]:
                continue
            groups[v].append(im)
        rows = []
        for v, frames in sorted(groups.items()):
            # Prefer an annotated sign large enough to inspect, across independent source videos.
            def rank(im):
                area = max(o["area"] for o in anns[im["id"]])
                return (abs(area-1700), im["file_name"])
            for im in sorted(frames,key=rank):
                data=z.read(prefix+im["file_name"])
                img=Image.open(BytesIO(data))
                fingerprint=dhash(img)
                if any((fingerprint^old).bit_count()<=6 for old in fingerprints):
                    continue
                fingerprints.append(fingerprint)
                w,h=img.size
                name="vn4_"+hashlib.sha256(data).hexdigest()[:18]+".jpg"
                boxes=[]
                ann=ET.Element("annotation")
                for obj in anns[im["id"]]:
                    x,y,bw,bh=obj["bbox"]
                    if bw<=0 or bh<=0:
                        continue
                    b=[x,y,x+bw,y+bh]
                    o=ET.SubElement(ann,"object")
                    ET.SubElement(o,"name").text="traffic_sign"
                    bounds=ET.SubElement(o,"bndbox")
                    for key,value in zip(("xmin","ymin","xmax","ymax"),b):
                        ET.SubElement(bounds,key).text=str(value)
                    boxes.append(dict(original_class_id=obj["category_id"],xyxy=b))
                rows.append(dict(file=name,source=prefix+im["file_name"],video=v,original_size=[w,h],size=[w,h],
                                 image_sha256=hashlib.sha256(data).hexdigest(),boxes=boxes))
                result.writestr("images/"+name,data)
                result.writestr("source_xml/"+Path(name).stem+".xml",ET.tostring(ann))
                break
            if len(rows)==36:
                break
        result.writestr("manifest.json",json.dumps({"source":"Vietnam Traffic Sign Detection v5, Roboflow (train split)",
                        "excluded_videos":sorted(x for x in prior if x), "records":rows},indent=1))
        print(len(rows),"images,",sum(len(r["boxes"]) for r in rows),"boxes;",len(groups),"eligible video groups")


if __name__=="__main__":
    main()
