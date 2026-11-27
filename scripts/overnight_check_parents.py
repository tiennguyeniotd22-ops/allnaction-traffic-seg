"""Check full parents and reviewed crops against the fixed splits and prior sources."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from overnight_collect_coco import dhash
from overnight_dataset import sha
from overnight_scenes import extract


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("candidates",type=Path,nargs="+")
    p.add_argument("--base-version",default="e2-v1")
    p.add_argument("--previous-extension", action="append", type=Path, default=[])
    a=p.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(26)
    refs={}; groups={}; retained={}; by_sha={}; exact=[]
    def add(name, group, content):
        digest=sha(content)
        if digest in by_sha:
            original=by_sha[digest]
            if group!=groups[original] and group!="base":
                exact.append([group,groups[original],name,original,"sha256"])
            return
        by_sha[digest]=name
        refs[name]=extract(content)
        groups[name]=group
        retained[name]=dhash(Image.open(BytesIO(content)))
    for file in ("dataset","holdout"):
        with ZipFile(a.root/(a.base_version if file=="dataset" else "e2-v1")/(file+".zip")) as z:
            for name in z.namelist():
                if name.startswith("images/") and Path(name).suffix.lower() in (".jpg",".png",".jpeg"):
                    add(file+"/"+name,"base",z.read(name))
    previous=json.loads((a.root/"e2-v1/manifest.json").read_text())["records"]
    for r in previous:
        key=r["key"]
        add("e2-parent/"+key,"base",(a.root/r["source"]/"images"/(key+".jpg")).read_bytes())
    for previous_manifest in a.previous_extension:
        for r in json.loads(previous_manifest.read_text())["records"]:
            data=(a.root/r["parent"]).read_bytes()
            assert sha(data)==r["parent_sha256"]
            add("prior-parent/"+r["key"],"base",data)
    for candidate in a.candidates:
        for r in json.loads((candidate/"manifest.json").read_text()):
            key=r["key"]
            parent=(a.root/r["parent"]).read_bytes()
            child=(candidate/"images"/(key+".jpg")).read_bytes()
            assert sha(child)==r["image_sha256"] and sha(parent)==r["parent_sha256"]
            add(key+"/parent",key,parent)
            add(key+"/crop",key,child)
    names=[n for n in refs if len(refs[n][1])>=2]
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=92))
    matcher.add([refs[n][1] for n in names])
    matcher.train()
    local=cv2.BFMatcher()
    results=[]; checked=set()
    for name in groups:
        if groups[name]=="base":continue
        fp=retained[name]
        for other,other_hash in retained.items():
            if other!=name and groups[name]!=groups[other] and (fp^other_hash).bit_count()<=4:
                exact.append([groups[name],groups[other],name,other,"dHash<=4"])
        xy,d=refs[name]
        if len(d)<9:continue
        hits=Counter()
        for group in matcher.knnMatch(d,k=10):
            hits.update(set(names[m.imgIdx] for m in group if groups[names[m.imgIdx]]!=groups[name] and m.distance<260))
        for other,votes in hits.most_common(28):
            pair=tuple(sorted((name,other)))
            if votes<6 or pair in checked:continue
            checked.add(pair)
            points,desc=refs[other]
            good=[s for s,t in local.knnMatch(d,desc,k=2) if s.distance<.76*t.distance]
            if len(good)<9:continue
            x=np.float32([xy[m.queryIdx] for m in good])
            y=np.float32([points[m.trainIdx] for m in good])
            _,mask=cv2.findHomography(x,y,cv2.RANSAC,3.2)
            if mask is None:continue
            inliers=mask.ravel().astype(bool)
            if sum(inliers)>=10 and min(cv2.contourArea(cv2.convexHull(v[inliers])) for v in (x,y))>=520:
                results.append(dict(a=groups[name],b=groups[other],left=name,right=other,inliers=int(sum(inliers))))
    report=dict(base_dataset_sha256=sha((a.root/a.base_version/"dataset.zip").read_bytes()),holdout_sha256=sha((a.root/"e2-v1/holdout.zip").read_bytes()),
                candidate_manifest_sha256={folder.name:sha((folder/"manifest.json").read_bytes()) for folder in a.candidates},
                descriptor_images=len(names),checked=len(checked),close_hash=exact,same_scene=results)
    a.out.write_text(json.dumps(report,indent=1))
    print(len(names),len(checked),len(exact),len(results),flush=True)


if __name__=="__main__":
    main()
