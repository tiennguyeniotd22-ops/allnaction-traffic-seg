"""Generate COCO proposals on CPU for source-guided full-frame annotation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(3)
    model = YOLO("yolo26x-seg.pt")
    rows = {}
    for image in sorted(args.images.glob("*.jpg")):
        result = model.predict(str(image), imgsz=960, conf=0.20, device="cpu",
                               classes=[0, 1, 2, 3, 5, 7, 9, 11], verbose=False)[0]
        boxes = result.boxes
        rows[image.name] = [{"label": result.names[int(c)], "score": round(float(s), 4),
                             "box": b, "polygon": poly.tolist()}
                            for c, s, b, poly in zip(boxes.cls, boxes.conf, boxes.xyxy.tolist(),
                                                     result.masks.xy if result.masks else [], strict=True)]
        print(image.name, len(rows[image.name]), flush=True)
    args.output.write_text(json.dumps({"teacher_sha256": hashlib.sha256(Path("yolo26x-seg.pt").read_bytes()).hexdigest(),
                                       "images": rows}))

if __name__ == "__main__":
    main()
