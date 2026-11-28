"""Create reviewable two-wheeler cutouts and train-only composites on native road surfaces."""
import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import random
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from pycocotools import mask as mu

from overnight_dataset import sha, write_member
from overnight_drafts import COLORS, to_rle
from overnight_extend import check_labels


def read(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_bytes((json.dumps(data, indent=1) + "\n").encode())


def labels(body, size):
    for line in body.decode().splitlines():
        fields = line.split()
        yield int(fields[0]), np.array(fields[1:], dtype=float).reshape(-1, 2) * size


def sheets(rows, out, preview, prefix, cols=5, rows_per_page=4):
    for start in range(0, len(rows), cols * rows_per_page):
        sheet = Image.new("RGB", (cols * 232, rows_per_page * 174), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(rows[start:start+cols*rows_per_page]):
            im = preview(row)
            im.thumbnail((230, 150), Image.Resampling.LANCZOS)
            x, y = j % cols * 232, j // cols * 174
            sheet.paste(im, (x + (230-im.width)//2, y + (150-im.height)//2))
            draw.text((x+2, y+151), f"{start+j:03d}   cls {row['cls']}", fill="black")
        sheet.save(out / f"{prefix}-{start:03d}.jpg", quality=93)


def donors(root, out, archive):
    rows = []
    for version in ("team21-context-v1", "team21-vistas-extra-v1", "team21-road-v1", "e3-vistas-v1"):
        for row in read(root/version/"manifest.json")["records"]:
            stem = row.get("stem", "zz_plus_" + row["key"])
            source = "images/train/" + stem + ".jpg"
            if source not in archive.NameToInfo:
                continue
            body = archive.read(source)
            annotation = archive.read("labels/train/" + stem + ".txt")
            if sha(body) != row["image_sha256"] or sha(annotation) != row["label_sha256"]:
                continue
            image = Image.open(BytesIO(body)).convert("RGB")
            w, h = image.size
            for n, (cls, points) in enumerate(labels(annotation, (w,h))):
                if cls not in (4, 5, 6, 7):
                    continue
                l, t = np.floor(points.min(0)).astype(int)
                r, b = np.ceil(points.max(0)).astype(int)
                if min(l,t,w-r,h-b) < 3 or min(r-l,b-t) < max(25, .035*max(w,h)):
                    continue
                ring = points - [l,t]
                mask = mu.decode(to_rle(ring,r-l,b-t))*255
                if mask.mean()/255 < .13:
                    continue
                key = f"{stem}-{n}"
                piece = image.crop((l,t,r,b)).convert("RGBA")
                pad = max(12,round(max(r-l,b-t)*.4))
                around = np.asarray(image.crop((max(0,l-pad),max(0,t-pad),min(w,r+pad),min(h,b+pad))).convert("L"))
                piece.putalpha(Image.fromarray(mask))
                piece.save(out/(key+".png"))
                rows.append(dict(key=key, cls=cls, source=source, image_sha256=sha(body),
                                 label_sha256=sha(annotation), source_version=version, row=n,
                                 poly=ring.round(6).tolist(), size=list(piece.size), scene_luma=float(np.median(around))))
    assert len({r["key"] for r in rows}) == len(rows)
    rows.sort(key=lambda r: (r["cls"], sha(("21/"+r["key"]).encode())))
    write_json(out/"donors.json", rows)

    def preview(row):
        im = Image.open(out/(row["key"]+".png"))
        base = Image.new("RGB", im.size, "#9a9a9a")
        base.paste(im, mask=im.getchannel("A"))
        return base
    sheets(rows, out, preview, "donor", cols=7, rows_per_page=5)
    print("Cutouts", len(rows), Counter(r["cls"] for r in rows))


def render(root, out, archive, accepted):
    bank = read(out/"donors.json")
    picks = read(accepted)
    assert picks["donors_sha256"] == sha((out/"donors.json").read_bytes())
    by_class = defaultdict(list)
    for i in sorted(set(picks["keep"])):
        by_class[bank[i]["cls"]].append(bank[i])
    rng = random.Random(21)
    selected = read(root/"team21-vistas-extra-v1/manifest.json")["records"]
    idx = read(root/"team21-mapillary-extra/index.json")
    categories = idx["categories"]
    originals = {r["key"]: r for r in idx["records"]}
    ground_names = {"bike-lane","parking","parking-aisle","road","road-shoulder","service-lane","sidewalk"}
    ground_class = {int(k) for k, v in categories.items() if v.removeprefix("construction--flat--") in ground_names}
    assert {13,17,18,21,22,23,24} == ground_class
    outputs, counts = [], Counter()
    usage = Counter()
    for row in sorted(selected, key=lambda r: sha(("21/bg/"+r["key"]).encode())):
        stem = row["stem"]
        bg = archive.read("images/train/"+stem+".jpg")
        body = archive.read("labels/train/"+stem+".txt")
        assert sha(bg) == row["image_sha256"] and sha(body) == row["label_sha256"]
        image = Image.open(BytesIO(bg)).convert("RGB")
        w,h = image.size
        occupied = np.zeros((h,w), np.uint8)
        anchors = []
        for c, pts in labels(body, image.size):
            occupied |= mu.decode(to_rle(pts,w,h))
            if c in (0,1,2,4,5,6,7,8):
                l,t,r,b = *pts.min(0), *pts.max(0)
                if .045*h < b-t < .70*h and 0.02*h < b < .97*h:
                    # Infer physical projection from a road user with a visible base.
                    height_metres = {0:1.5,1:3.0,2:2.8,4:1.1,5:1.55,6:1.0,7:1.55,8:1.7}[c]
                    anchors.append((b,(b-t)/height_metres))
        if not anchors:
            continue
        native = root/"team21-mapillary-extra/native"/(row["key"]+".png")
        assert sha(native.read_bytes()) == row["source_mask_sha256"]
        parent = originals[row["key"]]
        assert parent["source_split"] == "training"
        ids = [s["id"] for s in parent["segments"] if s["category_id"] in ground_class]
        rgb = np.asarray(Image.open(native).convert("RGB").crop(row["roi"]).resize((w,h), Image.Resampling.NEAREST), np.int32)
        ground = np.isin(rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2],ids)
        inside_ground = cv2.distanceTransform(ground.astype(np.uint8),cv2.DIST_L2,3)
        for variant in range(3):
            cls = min(range(4,8), key=lambda c:(counts[c],rng.random()))
            choices = [d for d in by_class[cls] if Path(d["source"]).stem != stem]
            rng.shuffle(choices)
            choices.sort(key=lambda d: usage[d["key"]])
            if not choices:
                continue
            placed = False
            for donor in choices[:6]:
                im = Image.open(out/(donor["key"]+".png")).convert("RGBA")
                dw, dh = im.size
                for _ in range(100):
                    foot, ppm = rng.choice(anchors)
                    height = round(ppm * (1.5 if cls in (5,7) else 1.05) * rng.uniform(.82,1.18))
                    width = round(height * dw/dh)
                    if height < .04*max(w,h) or height > .43*h or min(width,height)<16 or width>w//2:
                        continue
                    bottom = round(foot+rng.uniform(-.025,.025)*h)
                    if bottom <= height or bottom >= h-3:
                        continue
                    left = rng.randrange(2,w-width-1)
                    top = bottom-height
                    under = ground[bottom-2:bottom+3, left+width//8:left+width-width//8]
                    if under.size == 0 or under.mean() < .94:
                        continue
                    if cls in (4,6) and inside_ground[bottom,left+width//2] > max(10,.25*height):
                        continue
                    layer = im.resize((width,height),Image.Resampling.LANCZOS)
                    poly = np.array(donor["poly"])*[width/dw,height/dh]+[left,top]
                    local = mu.decode(to_rle(poly-[left,top],width,height)).astype(bool)
                    if np.any(local & occupied[top:bottom,left:left+width].astype(bool)):
                        continue
                    alpha = np.asarray(layer.getchannel("A"))>=128
                    if (alpha & local).sum()/max(1,(alpha|local).sum()) < .94:
                        continue
                    final = image.copy()
                    margin = max(12,round(max(width,height)*.4))
                    vicinity = image.crop((max(0,left-margin),max(0,top-margin),min(w,left+width+margin),min(h,bottom+margin)))
                    exposure = float(np.clip((np.median(np.asarray(vicinity.convert("L")))+12)/(donor["scene_luma"]+12),.65,1.30))
                    alpha_image = layer.getchannel("A")
                    layer = ImageEnhance.Brightness(layer.convert("RGB")).enhance(exposure)
                    # An ambient contact shadow is restricted to the annotated ground surface.
                    y0,y1 = max(0,bottom-round(height*.12)-2),min(h,bottom+round(height*.12)+3)
                    xs = np.arange(width)-(width-1)*.5
                    ys = np.arange(y0,y1)-bottom
                    shadow = .28*np.exp(-.5*((ys[:,None]/max(1,height*.028))**2+(xs[None,:]/max(2,width*.28))**2))
                    shadow *= ground[y0:y1,left:left+width]
                    patch = np.asarray(final.crop((left,y0,left+width,y1)))*(1-shadow[:,:,None])
                    final.paste(Image.fromarray(patch.astype(np.uint8)),(left,y0))
                    final.paste(layer,(left,top),alpha_image)
                    key = f"paste21_{row['key']}_{variant}"
                    line = str(cls)+" "+" ".join(f"{v:.7f}" for v in (poly/[w,h]).ravel())+"\n"
                    result = body.rstrip()+b"\n"+line.encode()
                    check_labels(result, [dict(cls=c) for c,_ in labels(result,(w,h))])
                    jpg = out/(key+".jpg")
                    final.save(jpg,quality=94,subsampling=0)
                    (out/(key+".txt")).write_bytes(result)
                    outputs.append(dict(key=key,cls=cls,background=row["key"],background_sha256=sha(bg),
                        background_label_sha256=sha(body),donor=donor["key"],donor_source_sha256=donor["image_sha256"],
                        box=[left,top,left+width,bottom],exposure=round(exposure,5),
                        image_sha256=sha(jpg.read_bytes()),label_sha256=sha(result)))
                    counts[cls]+=1
                    usage[donor["key"]]+=1
                    placed=True
                    break
                if placed:
                    break
    write_json(out/"composites.json",outputs)

    def preview(row):
        im = Image.open(out/(row["key"]+".jpg"))
        l,t,r,b = row["box"]
        delta=max(20,round(max(r-l,b-t)*.85))
        return im.crop((max(0,l-delta),max(0,t-delta),min(im.width,r+delta),min(im.height,b+delta)))
    sheets(outputs,out,preview,"scene",cols=6,rows_per_page=5)
    print("Placed",len(outputs),counts,len(usage),"donors")


def pack(root, out, archive, accepted):
    rows = read(out/"composites.json")
    picks = read(accepted)
    assert picks["composites_sha256"] == sha((out/"composites.json").read_bytes())
    keep = [rows[n] for n in sorted(set(picks["keep"]))]
    version=root/"team21-paste-v1"
    version.mkdir(exist_ok=True)
    target=version/"dataset.zip"
    if target.exists() or not keep:
        raise ValueError("Expected a new destination and at least one reviewed composite")
    temp=target.with_suffix(".partial")
    with ZipFile(temp,"w") as z:
        for name in archive.namelist():
            write_member(z,name,archive.read(name))
        for row in keep:
            for sub,ext in (("images","jpg"),("labels","txt")):
                key=row["key"]+"."+ext
                data=(out/key).read_bytes()
                assert sha(data)==row["image_sha256" if ext=="jpg" else "label_sha256"]
                write_member(z,sub+"/train/"+key,data)
    with ZipFile(temp) as z:
        assert z.testzip() is None
        val=read(root.parent/"audit-phase12/validation-manifest.json")
        assert len(val)==140
        for row in val:
            name=row["file"]
            assert sha(z.read("images/val/"+name))==row["image_sha256"]
            assert sha(z.read("labels/val/"+Path(name).stem+".txt"))==row["labels_sha256"]
    temp.replace(target)
    manifest=dict(train=1693+len(keep),val=140,seed=21,added_images=len(keep),
        source="Derived from reviewed training parents in the referenced version manifests",
        base_sha256=sha(Path(archive.filename).read_bytes()),dataset_sha256=sha(target.read_bytes()),
        pasted_by_class=dict(Counter(r["cls"] for r in keep)),selection_sha256=sha(accepted.read_bytes()),records=keep)
    write_json(version/"manifest.json",manifest)
    print(manifest["train"],manifest["pasted_by_class"],manifest["dataset_sha256"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["donors","render","pack"])
    parser.add_argument("root",type=Path)
    parser.add_argument("--selection",type=Path)
    args=parser.parse_args()
    out=args.root/"team21-paste-review"
    out.mkdir(exist_ok=True)
    source=args.root/"team21-vistas-extra-v1/dataset.zip"
    assert sha(source.read_bytes())=="5e3460b7d794fdea49ea0654870ba60d26deb093907141107972717b430ab371"
    with ZipFile(source) as z:
        if args.mode=="donors":
            donors(args.root,out,z)
        else:
            if not args.selection:
                parser.error("A review selection is required")
            (render if args.mode=="render" else pack)(args.root,out,z,args.selection)


if __name__=="__main__":
    main()
