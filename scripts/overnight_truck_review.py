"""Render each train truck polygon with context and an immutable source-line identifier."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from overnight_dataset import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    index = []
    cols, rows, width, height = 5, 5, 222, 130
    with ZipFile(a.dataset) as z:
        for name in sorted(n for n in z.namelist() if n.startswith("labels/train/") and n.endswith(".txt")):
            content = z.read(name)
            lines = content.decode().splitlines()
            truck = [i for i, line in enumerate(lines) if line.split()[0] == "2"]
            if not truck:
                continue
            image_file = "images/train/" + Path(name).stem + ".jpg"
            body = z.read(image_file)
            im = Image.open(BytesIO(body)).convert("RGB")
            for i in truck:
                pts = np.asarray(lines[i].split()[1:], float).reshape(-1, 2) * im.size
                left, top = pts.min(axis=0)
                right, bottom = pts.max(axis=0)
                mx, my = max(10, .24 * (right - left)), max(8, .3 * (bottom - top))
                box = (max(0, int(left-mx)), max(0, int(top-my)), min(im.width, int(right+mx+1)), min(im.height, int(bottom+my+1)))
                view = im.crop(box)
                view.thumbnail((width, height))
                coords = (pts - box[:2]) * [view.width / (box[2]-box[0]), view.height / (box[3]-box[1])]
                points = [tuple(x) for x in coords]
                ImageDraw.Draw(view).line(points + [points[0]], fill="#ffad51", width=1)
                key = len(index)
                if key % (cols*rows) == 0:
                    sheet = Image.new("RGB", (cols*width, rows*(height+17)), "#eeeeee")
                    d = ImageDraw.Draw(sheet)
                x = key % cols * width
                y = (key // cols) % rows * (height+17)
                sheet.paste(view, (x,y))
                d.text((x+1,y+height), str(key), fill="#000000")
                index.append(dict(key=key,file=name,line=i,image_sha256=sha(body),label_sha256=sha(content),
                                  polygon_sha256=sha(lines[i].encode()),source_box=box))
                if (key+1) % (cols*rows) == 0:
                    sheet.save(a.out/f"{key//(cols*rows):03}.jpg", quality=92)
        if index and len(index) % (cols*rows):
            sheet.save(a.out/f"{(len(index)-1)//(cols*rows):03}.jpg", quality=92)
    (a.out/"index.json").write_text(json.dumps(dict(dataset_sha256=sha(a.dataset.read_bytes()),records=index),indent=1))
    print(len(index))


if __name__ == "__main__":
    main()
