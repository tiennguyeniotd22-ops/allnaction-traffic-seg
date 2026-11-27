"""Render indexed overlays or source crops in a stable review order."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--width",type=int,default=285);p.add_argument("--height",type=int,default=158)
    p.add_argument("--cols",type=int,default=4);p.add_argument("--rows",type=int,default=5)
    a=p.parse_args()
    files=sorted(a.source.glob("*.jpg"));n=a.rows*a.cols
    a.out.mkdir(exist_ok=True,parents=True)
    for start in range(0,len(files),n):
        sheet=Image.new("RGB",(a.width*a.cols,(a.height+18)*a.rows),"#ebebeb");d=ImageDraw.Draw(sheet)
        for j,f in enumerate(files[start:start+n]):
            im=Image.open(f);im.thumbnail((a.width-2,a.height))
            x=j%a.cols*a.width;y=j//a.cols*(a.height+18)
            sheet.paste(im,(x,y));d.text((x+2,y+a.height+1),f.stem,fill="black")
        sheet.save(a.out/f"{start:04}.jpg",quality=91)
    print(len(files),flush=True)


if __name__=="__main__":
    main()
