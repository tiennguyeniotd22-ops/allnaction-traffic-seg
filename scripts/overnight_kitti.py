"""Stage the 200 publicly labeled KITTI 2015 training images and original instances."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from overnight_source_archive import SourceArchive

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    a=SourceArchive(URL)
    files=sorted(n for n in a.archive.namelist() if n.startswith("training/image_2/") and n.endswith("_10.png"))
    assert len(files)==200
    for sub in ("images","native"):
        (args.out/sub).mkdir(exist_ok=True,parents=True)
    def fetch(name):
        key="kt"+Path(name).stem.split("_")[0]
        record=dict(key=key,image=name,annotation=name.replace("image_2","instance"),split="training")
        for keyname,sub in (("image","images"),("annotation","native")):
            f=args.out/sub/(key+".png")
            data=a.member(record[keyname])
            f.write_bytes(data)
            record[keyname+"_sha256"]=hashlib.sha256(data).hexdigest()
        return record
    with ThreadPoolExecutor(6) as executor:
        rows=list(executor.map(fetch,files))
    report=dict(source="KITTI Vision Benchmark Suite, 2015 instance segmentation",source_archive=URL,
                source_page="https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015",
                license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0",records=rows)
    (args.out/"index.json").write_text(json.dumps(report,indent=1))
    print("staged",len(rows),flush=True)


if __name__=="__main__":
    main()
