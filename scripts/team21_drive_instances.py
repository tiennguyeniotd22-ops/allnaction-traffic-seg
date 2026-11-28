"""Render individual SAM masks with their visual context for review before positive-only training."""
from collections import defaultdict
from pathlib import Path
import argparse,json
from PIL import Image, ImageDraw
import numpy as np
from pycocotools import mask as coco
from overnight_drafts import to_rle

def main():
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    root=p.parse_args().folder
    source=root/"curated"
    rows=json.loads((source/"manifest.json").read_bytes())
    out=root/"instances"
    out.mkdir(exist_ok=True)
    labels=[]
    patches=[]
    for row in rows:
        key=row["key"]
        im=Image.open(root/"parents"/(key+".png")).convert("RGB")
        w,h=im.size
        by_cls=defaultdict(list)
        body=(source/"labels"/(key+".txt")).read_text().splitlines()
        for obj,line in zip(row["objects"],body,strict=True):
            c=obj["cls"]
            pts=np.array(line.split()[1:],float).reshape(-1,2)*[w,h]
            lo=pts.min(0)
            hi=pts.max(0)
            by_cls[c].append(((hi-lo).prod(),obj,pts))
        for c,items in sorted(by_cls.items()):
            limit=5 if c in (4,5,6,7) else 2
            for area,obj,pts in sorted(items,key=lambda s:-s[0])[:limit]:
                x,y=pts.min(0); r,b=pts.max(0)
                margin=max(12,.24*max(r-x,b-y))
                crop=(max(0,int(x-margin)),max(0,int(y-margin)),min(w,int(r+margin+1)),min(h,int(b+margin+1)))
                native=coco.decode(to_rle(pts,w,h)).astype(bool)
                highlight=np.asarray(im).copy()
                highlight[native]=.54*highlight[native]+.46*np.array([255,82,40])
                panel=Image.new("RGB",(182,97),"#d8d8d8")
                for n,img in enumerate((im,Image.fromarray(highlight))):
                    v=img.crop(crop)
                    v.thumbnail((90,94),Image.Resampling.LANCZOS)
                    panel.paste(v,(n*91+(91-v.width)//2,(96-v.height)//2))
                patches.append(panel)
                labels.append(dict(key=key,row=obj["row"],cls=c))
    for start in range(0,len(labels),40):
        chunk=patches[start:start+40]
        sheet=Image.new("RGB",(5*184,((len(chunk)+4)//5)*116),"white")
        draw=ImageDraw.Draw(sheet)
        for j,img in enumerate(chunk):
            i=start+j
            x=j%5*184;y=j//5*116
            sheet.paste(img,(x,y))
            row=labels[i]
            draw.text((x+2,y+98),f"{i:03d}  #{row['key'][-3:]}  cls {row['cls']}",fill="black")
        sheet.save(out/f"{start:03}.jpg",quality=93)
    (out/"index.json").write_text(json.dumps(labels,indent=1))
    print(len(labels))

if __name__=="__main__":
    main()
