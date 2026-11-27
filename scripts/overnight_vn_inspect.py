"""Lay out each SAM prompt beside its visible mask and original street scene."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path);p.add_argument("masks",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    with ZipFile(a.source) as source,ZipFile(a.masks) as archive:
        records=json.loads(archive.read("source-objects.json"))
        assert archive.testzip() is None
        for start in range(0,len(records),9):
            page=Image.new("RGB",(8*142,9*116),"#ededed")
            canvas=ImageDraw.Draw(page)
            for j,name in enumerate(sorted(records)[start:start+9]):
                frame=Image.open(BytesIO(source.read("images/"+name))).convert("RGB")
                original=frame.copy();original.thumbnail((140,94))
                y=j*116;page.paste(original,(0,y))
                canvas.text((2,y+95),name[:-4],fill="black")
                for obj in records[name][:7]:
                    idx=obj["index"]
                    m=np.asarray(Image.open(BytesIO(archive.read(obj["mask"]))).convert("L"))>0
                    l,t,r,b=obj["box"]
                    w,h=frame.size
                    border=max(8,.23*max(r-l,b-t))
                    roi=(int(max(0,l-border)),int(max(0,t-border)),int(min(w,r+border)),int(min(h,b+border)))
                    tile=frame.crop(roi)
                    raster=m[roi[1]:roi[3],roi[0]:roi[2]]
                    scale=min(140/tile.width,94/tile.height)
                    size=(max(1,round(tile.width*scale)),max(1,round(tile.height*scale)))
                    tile=tile.resize(size,Image.Resampling.LANCZOS)
                    small=cv2.resize(raster.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
                    contours,_=cv2.findContours(small,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    data=np.array(tile)
                    cv2.drawContours(data,contours,-1,(65,255,130),1)
                    x=(idx+1)*142
                    page.paste(Image.fromarray(data),(x,y))
                    canvas.text((x+2,y+95),f"{idx} q={obj['quality']:.2f}",fill="black")
            page.save(a.out/f"{start:03}.jpg",quality=92)
    print(len(records),flush=True)


if __name__=="__main__":
    main()
