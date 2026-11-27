"""Segment each original Poribohon road-vehicle box using CVAT's frozen SAM encoder and decoder."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--checkpoint", default="/opt/nuclio/sam/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=args.checkpoint).cpu().eval())
    output = args.output.with_suffix(".partial")
    if args.output.exists():
        raise FileExistsError(args.output)
    with ZipFile(args.source) as source, ZipFile(output, "w", ZIP_DEFLATED) as archive, torch.inference_mode():
        result = {}
        for row in json.loads(source.read("manifest.json"))["records"]:
            name = row["file"]
            w, h = row["size"]
            ow, oh = row["original_size"]
            xml = ET.fromstring(source.read("source_xml/"+Path(name).stem+".xml"))
            prompts = []
            for j, obj in enumerate(xml.findall("object")):
                label = obj.findtext("name")
                if label in {"boat", "launch"}:
                    continue
                box = [float(obj.findtext("bndbox/"+k)) * scale for k, scale in
                       zip(("xmin","ymin","xmax","ymax"), (w/ow,h/oh,w/ow,h/oh))]
                box = np.clip(box,[0,0,0,0],[w,h,w,h]).tolist()
                if box[0] < box[2] and box[1] < box[3]:
                    prompts.append({"index":j, "source_label":label, "box":box})
            binary = source.read("images/"+name)
            request = Request(args.endpoint, data=json.dumps({"image":base64.b64encode(binary).decode()}).encode(),
                              headers={"Content-Type":"application/json"})
            with urlopen(request, timeout=120) as response:
                embedding = base64.b64decode(json.load(response)["blob"])
            predictor.reset_image()
            predictor.original_size = (h,w)
            predictor.input_size = predictor.transform.get_preprocess_shape(h,w,1024)
            predictor.features = torch.from_numpy(np.frombuffer(embedding,np.float32).copy().reshape(1,256,64,64))
            predictor.is_image_set = True
            for start in range(0,len(prompts),4):
                batch = prompts[start:start+4]
                boxes = torch.tensor([b["box"] for b in batch],dtype=torch.float32)
                transformed = predictor.transform.apply_boxes_torch(boxes,(h,w))
                masks,quality,_ = predictor.predict_torch(None,None,boxes=transformed,multimask_output=False)
                for j,obj in enumerate(batch):
                    buf = BytesIO()
                    Image.fromarray(masks[j,0].numpy()).save(buf,format="PNG")
                    obj["mask"] = f"masks/{Path(name).stem}-{obj['index']:03}.png"
                    obj["quality"] = float(quality[j,0])
                    archive.writestr(obj["mask"],buf.getvalue())
            result[name] = prompts
            print(name,len(prompts),flush=True)
        archive.writestr("source-objects.json",json.dumps(result))
    output.rename(args.output)

if __name__ == "__main__":
    main()
