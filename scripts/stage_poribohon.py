"""Stage the authors' mixed-scene Poribohon-BD sample for mask review."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image


def dhash(image):
    grey = np.asarray(image.convert("L").resize((9,8),Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(grey[:,1:] > grey[:,:-1]).tobytes(), "big")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sample", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    known = []
    with ZipFile(args.baseline) as base:
        for n in base.namelist():
            if n.startswith("images/"):
                with Image.open(BytesIO(base.read(n))) as img:
                    known.append(dhash(img))
    records = []
    counts = Counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(".partial")
    with ZipFile(args.sample) as source, ZipFile(temporary, "w", ZIP_DEFLATED) as out:
        paths = {Path(n).stem.lower(): n for n in source.namelist()
                 if "/images/" in n and Path(n).suffix.lower() in {".png",".jpg",".jpeg"}}
        for name in sorted(n for n in source.namelist() if n.endswith(".xml")):
            xml = source.read(name)
            ann = ET.fromstring(xml)
            targets = [o for o in ann.findall("object") if o.findtext("name") in {"cng","rickshaw"}]
            if not targets:
                continue
            orig = Path(ann.findtext("filename")).name
            path = paths[Path(orig).stem.lower()]
            data = source.read(path)
            with Image.open(BytesIO(data)) as im:
                w,h = im.size
                if (w,h) != tuple(int(ann.findtext("size/"+k)) for k in ("width","height")):
                    raise ValueError(orig)
                fingerprint = dhash(im)
                if any((fingerprint ^ old).bit_count() <= 4 for old in known):
                    continue
                image = im.convert("RGB")
                image.thumbnail((1280,1280), Image.Resampling.LANCZOS)
            iw, ih = image.size
            boxes = []
            for obj in targets:
                box = [int(obj.findtext("bndbox/"+k)) for k in ("xmin","ymin","xmax","ymax")]
                x1, y1, x2, y2 = box
                if (x2-x1)*(y2-y1)*(640/max(w,h))**2 < 64:
                    continue
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    continue
                boxes.append(dict(source_label=obj.findtext("name"), original_xyxy=box, class_id=3,
                                  xyxy=[x1*iw/w,y1*ih/h,x2*iw/w,y2*ih/h]))
            if not boxes:
                continue
            buf = BytesIO()
            image.save(buf,format="JPEG",quality=94)
            resized = buf.getvalue()
            target = "bd_" + Path(orig).stem.replace(" (","_").replace(")","")+".jpg"
            out.writestr("images/"+target,resized)
            out.writestr("source_xml/"+Path(target).stem+".xml",xml)
            known.append(fingerprint)
            records.append(dict(file=target, source=path, original_sha256=hashlib.sha256(data).hexdigest(),
                                xml_sha256=hashlib.sha256(xml).hexdigest(), image_sha256=hashlib.sha256(resized).hexdigest(),
                                original_size=[w,h], size=[iw,ih], boxes=boxes))
            counts.update(b["source_label"] for b in boxes)
        report = dict(title="Poribohon-BD",doi="10.17632/pwyyg8zmk5.1",license="CC BY 4.0",
                      authors=["Shaira Tabassum","Md. Sabbir Ullah","Nakib Hossain Al-nur","Swakkhar Shatabda"],
                      origin="https://github.com/shairatabassum/BangladeshiVehiclesDataset/tree/d185287ebff01849089d560be0cc9f75305f89ee",
                      source_zip_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                      status="train candidate, full-frame independent QC pending",counts=counts,records=records)
        out.writestr("manifest.json",json.dumps(report,indent=1))
    temporary.rename(args.output)
    print(len(records),counts,hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
