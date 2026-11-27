"""Export editable class-3 draft polygons and keep every source mask in the SAM ZIP."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from shapely.geometry import Polygon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True)
    with ZipFile(args.input) as z:
        d = json.loads(z.read("manifest.json"))
        for row in d["records"]:
            (args.output/row["file"]).write_bytes(z.read("images/"+row["file"]))
            w,h=row["size"]
            lines=[]
            for obj in row["boxes"]:
                mask=np.array(Image.open(BytesIO(z.read(obj["mask_path"]))),dtype=np.uint8)
                assert mask.shape==(h,w)
                source=mu.encode(np.asfortranarray(mask))
                area=int(mu.area(source))*(640/max(w,h))**2
                obj["area640"]=area
                obj["draft_row"]=None
                if area<64:
                    obj["review_note"]="below evaluator area or empty SAM"
                    continue
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                largest=max(contours,key=cv2.contourArea)
                pts=cv2.approxPolyDP(largest,.65,True).reshape(-1,2)
                if len(pts)<3:
                    continue
                poly=(pts+.5)/[w,h]
                if not Polygon(poly).is_valid:
                    obj["review_note"]="manual contour required"
                    continue
                text=" ".join(f"{v:.8f}" for v in poly.ravel())
                parsed=np.fromstring(text,sep=" ").reshape(-1,2)*[w,h]
                # CVAT annotation coordinates are serialized with two decimal digits.
                rle=mu.frPyObjects([np.round(parsed,2).ravel().tolist()],h,w)[0]
                fit=float(mu.iou([rle],[source],[0])[0,0])
                obj["polygon_to_sam_iou"]=fit
                if fit<.93:
                    obj["review_note"]="lossy polygon; refer to full mask"
                    continue
                lines.append("3 "+text)
                obj["draft_row"]=len(lines)
            (args.output/(Path(row["file"]).stem+".txt")).write_text("\n".join(lines))
        (args.output/"review-manifest.json").write_text(json.dumps(d,indent=1))
        print("frames",len(d["records"]), "editable polygons",sum(b["draft_row"] is not None for r in d["records"] for b in r["boxes"]))


if __name__=="__main__":
    main()
