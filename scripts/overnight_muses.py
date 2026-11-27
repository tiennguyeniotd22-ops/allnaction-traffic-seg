"""Stage official MUSES train RGB and panoptic pairs by recording and condition."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import zlib

from overnight_source_archive import SourceArchive

ROOT="https://muses.ethz.ch/MUSES_packages/"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--count",type=int,default=640)
    a=p.parse_args()
    rgb=SourceArchive(ROOT+"frame_camera_trainvaltest.zip")
    labels=SourceArchive(ROOT+"gt_panoptic_trainval.zip")
    for sub in ("images","native"):
        (a.out/sub).mkdir(parents=True,exist_ok=True)
    (a.out/"License.pdf").write_bytes(labels.member("muses/License.pdf"))
    native=labels.member("muses/gt_panoptic/train.json")
    truth=json.loads(native)
    anno={r["image_id"]:r for r in truth["annotations"]}
    rng=random.Random(26)
    drives=sorted({r["id"].split("_")[0] for r in truth["images"]})
    rng.shuffle(drives)
    reserved=set(drives[:max(1,len(drives)//10)])
    groups=defaultdict(list)
    for row in truth["images"]:
        key=row["id"]
        rec=key.split("_")[0]
        split,weather,daytime,_=row["file_name"].split("/")
        if split!="train" or weather=="snow" or rec in reserved:
            continue
        objects=anno[key]["segments_info"]
        targets={o["category_id"] for o in objects if not o["iscrowd"] and o["area"]>=576}
        if not targets.intersection({19,20,24,25,26,27,28,32,33}):continue
        emphasis=len(targets.intersection({25,27,28,32,33}))+rng.random()*2.5
        groups[(weather,daytime,rec)].append((emphasis,row))
    items=[]
    keys=sorted(groups)
    rng.shuffle(keys)
    for group in keys:
        groups[group].sort(key=lambda x:-x[0])
        kept=[]
        for _,row in groups[group]:
            frame=int(row["id"].rsplit("_",1)[-1])
            if all(abs(frame-f)>=90 for f,_ in kept):kept.append((frame,row))
        groups[group]=[r for _,r in kept]
    for i in range(max(map(len,groups.values()))):
        items.extend(groups[k][i] for k in keys if i<len(groups[k]))
    items=items[:a.count]
    def fetch(row):
        key=row["id"]
        record=dict(key=key,drive=key.split("_")[0],condition="/".join(row["file_name"].split("/")[1:3]),
                    split="train",image="muses/frame_camera/"+row["file_name"],annotation="muses/gt_panoptic/"+row["gt_panoptic_file_name"],
                    segments=anno[key]["segments_info"])
        for ar,field,sub in [(rgb,"image","images"),(labels,"annotation","native")]:
            path=record[field]
            dest=a.out/sub/(key+".png")
            info=ar.archive.getinfo(path)
            if dest.is_file():
                data=dest.read_bytes()
                assert len(data)==info.file_size and zlib.crc32(data)==info.CRC
            else:
                data=ar.member(path)
                tmp=dest.with_suffix(".partial");tmp.write_bytes(data);tmp.replace(dest)
            record[field+"_sha256"]=hashlib.sha256(data).hexdigest()
        return record
    from concurrent.futures import as_completed
    records=[]
    errors={}
    with ThreadPoolExecutor(5) as pool:
        jobs={pool.submit(fetch,item):item["id"] for item in items}
        for future in as_completed(jobs):
            try:
                row=future.result()
            except OSError as err:
                errors[jobs[future]]=str(err)
                continue
            records.append(row)
            if len(records)%25==0:print("staged",len(records),flush=True)
    records.sort(key=lambda r:r["key"])
    report=dict(source="MUSES (ECCV 2024), ETH Zurich",homepage="https://muses.ethz.ch",rgb_zip=ROOT+"frame_camera_trainvaltest.zip",
                labels_zip=ROOT+"gt_panoptic_trainval.zip",license="ETH MUSES EULA, non-commercial research/education; models may be shared, images/masks may not",
                license_sha256=hashlib.sha256((a.out/"License.pdf").read_bytes()).hexdigest(),
                train_json_sha256=hashlib.sha256(native).hexdigest(),seed=26,min_frame_gap=90,reserved_drives=sorted(reserved),
                categories=truth["categories"],records=records,transport_failures=errors,requested=len(items))
    (a.out/"index.json").write_text(json.dumps(report,indent=1))
    print(len(records),Counter(r["condition"] for r in records),flush=True)


if __name__=="__main__":
    main()
