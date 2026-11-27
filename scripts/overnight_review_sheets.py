"""Render review sheets using retained file IDs and per-instance draft identifiers."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image, ImageDraw
import numpy as np
from overnight_drafts import draw, to_rle
from pycocotools import mask as mu


def base_review(zip_path,root):
    for sub in ("views","objects","images"):
        (root/sub).mkdir(parents=True,exist_ok=True)
    index=[]
    with ZipFile(zip_path) as z:
        names=sorted(n for n in z.namelist() if n.startswith("images/") and n.endswith(".jpg") and "zz_drive_" not in n)
        for i,n in enumerate(names):
            body=z.read(n)
            im=Image.open(BytesIO(body)).convert("RGB")
            w,h=im.size
            label=n.replace("images/","labels/").replace(".jpg",".txt")
            rows=z.read(label).decode().splitlines()
            objs=[]
            for j,line in enumerate(rows):
                vals=line.split()
                pts=np.asarray(vals[1:],float).reshape(-1,2)*[w,h]
                objs.append(dict(id="r"+str(j+1),cls=int(vals[0]),poly=pts.tolist(),
                                 area640=float(mu.area(to_rle(pts,w,h)))*(640/max(w,h))**2))
            key=f"team{i:03d}"
            (root/"images"/(key+".jpg")).write_bytes(body)
            (root/"objects"/(key+".json")).write_text(json.dumps(objs))
            draw(im,objs).save(root/"views"/(key+".jpg"),quality=90)
            index.append(dict(key=key,file=Path(n).name,split=n.split("/")[1],count=len(objs)))
    (root/"index.json").write_text(json.dumps(index,indent=1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--base",type=Path)
    p.add_argument("--cols",type=int,default=2)
    p.add_argument("--rows",type=int,default=2)
    p.add_argument("--width",type=int,default=580)
    p.add_argument("--height",type=int,default=360)
    p.add_argument("--plain",action="store_true")
    a=p.parse_args()
    if a.base:base_review(a.base,a.directory)
    files=sorted((a.directory/("images" if a.plain else "views")).glob("*.jpg"))
    out=a.directory/("contact-original" if a.plain else "contact")
    out.mkdir(exist_ok=True)
    n=a.cols*a.rows
    for k in range(0,len(files),n):
        canvas=Image.new("RGB",(a.cols*a.width,a.rows*(a.height+22)),"#ededed")
        d=ImageDraw.Draw(canvas)
        for j,f in enumerate(files[k:k+n]):
            im=Image.open(f);im.thumbnail((a.width-3,a.height))
            x=(j%a.cols)*a.width;y=(j//a.cols)*(a.height+22)
            canvas.paste(im,(x,y))
            d.text((x+3,y+a.height+2),f.stem,fill="black")
        canvas.save(out/f"{k:03d}.jpg",quality=90)
    print(len(files),out)


if __name__=="__main__":
    main()
