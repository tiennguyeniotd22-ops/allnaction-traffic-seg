"""Produce lossless SAM draft masks from Poribohon boxes for full-frame review."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(a.checkpoint)).eval().cuda())
    temporary = a.output.with_suffix(".partial")
    with ZipFile(a.input) as src, ZipFile(temporary,"w",ZIP_DEFLATED) as dst, torch.inference_mode():
        manifest = json.loads(src.read("manifest.json"))
        for row in manifest["records"]:
            path = "images/"+row["file"]
            data = src.read(path)
            dst.writestr(path,data)
            xml_name = "source_xml/"+Path(row["file"]).stem+".xml"
            dst.writestr(xml_name,src.read(xml_name))
            rgb = np.array(Image.open(BytesIO(data)).convert("RGB"))
            predictor.set_image(rgb)
            boxes = torch.tensor([b["xyxy"] for b in row["boxes"]],device="cuda",dtype=torch.float32)
            prompts = predictor.transform.apply_boxes_torch(boxes,rgb.shape[:2])
            for start in range(0,len(boxes),4):
                masks,quality,_ = predictor.predict_torch(None,None,boxes=prompts[start:start+4],multimask_output=False)
                for j, m in enumerate(masks[:,0].cpu().numpy()):
                    box = row["boxes"][start+j]
                    box["sam_quality_estimate"] = float(quality[j,0])
                    box["pixels"] = int(m.sum())
                    out = BytesIO()
                    Image.fromarray(m).save(out,format="PNG")
                    box["mask_path"] = f"masks/{Path(row['file']).stem}-{start+j:03}.png"
                    dst.writestr(box["mask_path"],out.getvalue())
            print(row["file"],len(boxes),flush=True)
        manifest["teacher"] = "SAM ViT-H, sam_vit_h_4b8939.pth, original VOC box prompts"
        dst.writestr("manifest.json",json.dumps(manifest,indent=1))
    temporary.rename(a.output)


if __name__ == "__main__":
    main()
