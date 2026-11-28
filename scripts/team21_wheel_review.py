"""Build the reviewed COCO wheel extension using original instance IDs for split decisions."""
import argparse
from collections import Counter
from pathlib import Path
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as masks

from overnight_dataset import encode, read, sha
from overnight_drafts import COLORS, to_rle
from overnight_extend import check_labels


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("plan",type=Path)
    a=p.parse_args()
    root=a.root
    folder=root/"team21-coco-frames-review"
    out=root/"team21-wheel-reviewed"
    for sub in ("images","labels","views"):
        (out/sub).mkdir(parents=True,exist_ok=True)
    plan=read(a.plan)
    assert plan["candidate_sha256"]==sha((folder/"manifest.json").read_bytes())
    rows=read(folder/"manifest.json")
    scene=read(root/"team21-coco-final-scenes.json")
    assert scene["candidate_manifest_sha256"][folder.name]==plan["candidate_sha256"]
    with ZipFile(root/"coco-more.zip") as z:
        truth={row["coco_id"]:row for row in json.loads(z.read("manifest.json"))}
    edges=[(r["a"],r["b"]) for r in scene["same_scene"]]+[r[:2] for r in scene["close_hash"]]
    kept=set()
    records=[]
    for index in sorted(set(plan["keep"])|{int(i) for i in plan["edit"]}):
        row=rows[index]
        key=row["key"]
        linked={b if a==key else a for a,b in edges if key in (a,b)}
        if linked.intersection({"base",*kept}):
            continue
        image=(folder/"images"/(key+".jpg")).read_bytes()
        label=(folder/"labels"/(key+".txt")).read_bytes()
        assert sha(image)==row["image_sha256"] and sha(label)==row["label_sha256"]
        w,h=row["size"]
        actions=plan["edit"].get(str(index),{})
        native={str(obj["id"]):obj for obj in truth[int(key[2:])]["objects"]}
        file=root/"coco-extra/objects"/(Path(row["parent"]).stem+".json")
        original={}
        for obj in read(file):
            for part in obj.get("components",[obj]):
                original[part["via"].split()[-1]]=part
        lines=[]
        objects=[]
        for obj,line in zip(row["objects"],label.decode().splitlines(),strict=True):
            ids=obj["native_ids"]
            if str(obj["cls"]) not in actions:
                lines.append(line)
                objects.append(obj)
                continue
            assert obj["cls"] in (5,7) and len(ids)>=2
            retained=set(actions[str(obj["cls"])])
            for i in ids:
                part=original[i]
                if part["cls"] not in retained:
                    continue
                xy=np.asarray(part["poly"])
                native_mask=masks.merge(masks.frPyObjects(native[i]["segmentation"],h,w))
                if np.ptp(xy,axis=0).min()<.0125*max(w,h):
                    continue
                line=encode(part["cls"],xy,w,h)
                serialized=np.asarray(line.split()[1:],float).reshape(-1,2)*[w,h]
                fidelity=float(masks.iou([to_rle(serialized,w,h)],[native_mask],[0])[0,0])
                if fidelity<.94:
                    raise ValueError((key,i,fidelity))
                lines.append(line)
                objects.append(dict(cls=part["cls"],native_ids=[i],iou=fidelity,area640=part["area640"]))
        body=("\n".join(lines)+"\n").encode()
        check_labels(body,objects)
        with Image.open(folder/"images"/(key+".jpg")) as source:
            view=source.convert("RGB")
        draw=ImageDraw.Draw(view)
        for obj,line in zip(objects,lines,strict=True):
            pts=list(map(tuple,np.asarray(line.split()[1:],float).reshape(-1,2)*[w,h]))
            draw.line(pts+pts[:1],fill=COLORS[obj["cls"]],width=2)
        view.save(out/"views"/(key+".jpg"),quality=92)
        (out/"images"/(key+".jpg")).write_bytes(image)
        (out/"labels"/(key+".txt")).write_bytes(body)
        records.append(dict(row,objects=objects,label_sha256=sha(body),prior_label_sha256=sha(label),review=actions))
        kept.add(key)
    (out/"manifest.json").write_bytes((json.dumps(records,indent=1)+"\n").encode())
    print(len(records),sum(bool(row["review"]) for row in records),Counter(o["cls"] for r in records for o in r["objects"]))


if __name__=="__main__":
    main()
