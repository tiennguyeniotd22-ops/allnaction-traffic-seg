"""Collect class-balanced road scenes from COCO train2017 with native instance masks."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZipFile, ZIP_DEFLATED
import requests
from PIL import Image
import numpy as np


def dhash(im):
    pixels=np.asarray(im.convert("L").resize((9,8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:,1:]>pixels[:,:-1]).tobytes(),"big")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("annotations",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--count",type=int,default=480)
    parser.add_argument("--previous",type=Path,help="Previously screened candidate ZIP to exclude by source image ID")
    a=parser.parse_args()
    cats={1:"person",2:"bicycle",3:"car",4:"motorcycle",6:"bus",8:"truck",10:"traffic light",13:"stop sign"}
    with ZipFile(a.annotations) as z:
        data=json.load(z.open("annotations/instances_train2017.json"))
        captions=json.load(z.open("annotations/captions_train2017.json"))
    licenses={r["id"]:r for r in data["licenses"]}
    images={r["id"]:r for r in data["images"] if r["license"] in (4,5)}
    street=set()
    tokens=("street","intersection","road","traffic","junction","sidewalk","crosswalk","parked","highway","motorcycl","bicycl","cyclist")
    for row in captions["annotations"]:
        if any(t in row["caption"].lower() for t in tokens):
            street.add(row["image_id"])
    del captions
    by=defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in cats and ann["image_id"] in images and ann["image_id"] in street:
            by[ann["image_id"]].append(ann)
    del data
    pools=defaultdict(list)
    for i,anns in by.items():
        if len(anns)>19 or any(x["iscrowd"] or not x["segmentation"] for x in anns):
            continue
        counts=Counter(x["category_id"] for x in anns)
        meta=images[i]
        plenty={c for c in counts if any(x["category_id"]==c and x["area"]*(640/max(meta["width"],meta["height"]))**2>=150 for x in anns)}
        if not plenty.difference({1}):
            continue
        for c in sorted(plenty-{1}):
            k=cats[c]
            if c in (2,4):
                k+= "-with-people" if counts[1] else "-unoccupied"
            pools[k].append(i)
    seed=random.Random(26)
    for k,items in sorted(pools.items()):
        items.sort()
        seed.shuffle(items)
    priority=["bicycle-unoccupied","motorcycle-unoccupied","bicycle-with-people","motorcycle-with-people","traffic light","stop sign","bus","truck","car"]
    selected=[]
    used=set()
    if a.previous:
        with ZipFile(a.previous) as previous:
            used.update(row["coco_id"] for row in json.loads(previous.read("manifest.json")))
    while len(selected)<a.count+40:
        before=len(selected)
        for k in priority:
            while pools[k] and pools[k][-1] in used:
                pools[k].pop()
            if pools[k]:
                i=pools[k].pop()
                used.add(i)
                selected.append((i,k))
        if len(selected)==before:break
    fingerprints=[]
    with ZipFile(a.reference) as old:
        for name in old.namelist():
            if name.startswith("images/") and name.endswith(".jpg"):
                fingerprints.append(dhash(Image.open(BytesIO(old.read(name)))))
    def fetch(item):
        i,k=item
        url=f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{i:012d}.jpg"
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=25);r.raise_for_status()
                im=Image.open(BytesIO(r.content))
                if im.size!=(images[i]["width"],images[i]["height"]):raise ValueError(i)
                return i,k,url,r.content,dhash(im)
            except requests.RequestException:
                if attempt==2:return i,k,url,None,None
    manifest=[]
    temp=a.output.with_suffix(".partial")
    if a.output.exists():raise FileExistsError(a.output)
    with ZipFile(temp,"w",ZIP_DEFLATED) as out, ThreadPoolExecutor(8) as pool:
        for i,k,url,body,fp in pool.map(fetch,selected):
            if body is None or any((fp^other).bit_count()<=4 for other in fingerprints):continue
            fingerprints.append(fp)
            name=f"coco5_{i:012d}.jpg"
            out.writestr("images/"+name,body)
            m=images[i]
            manifest.append(dict(file=name,coco_id=i,source_split="train2017",url=url,flickr_source=m.get("flickr_url"),
                license=licenses[m["license"]],size=[m["width"],m["height"]],sha256=sha256(body).hexdigest(),
                selection=k,objects=[dict(id=x["id"],label=cats[x["category_id"]],bbox=x["bbox"],segmentation=x["segmentation"]) for x in by[i]]))
            if len(manifest)>=a.count:break
        out.writestr("manifest.json",json.dumps(manifest))
    temp.rename(a.output)
    print(len(manifest),dict(Counter(r["selection"] for r in manifest)),flush=True)


if __name__=="__main__":
    main()
