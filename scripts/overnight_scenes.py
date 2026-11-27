"""Find cross-split scene matches using local visual features, including cropped views."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


def extract(data):
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    scale = 448 / max(im.shape)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, descriptors = cv2.SIFT_create(nfeatures=360).detectAndCompute(im, None)
    if descriptors is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    return np.float32([pt.pt for pt in points]), descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(26)
    refs = {}
    with ZipFile(args.base) as z:
        for name in sorted(z.namelist()):
            if name.startswith("images/") and name.endswith(".jpg"):
                refs[name] = extract(z.read(name))
    for source in ("bd","vn","coco"):
        for path in sorted((args.root/source/"images").glob("*.jpg")):
            refs[path.stem] = extract(path.read_bytes())
    names = [k for k in refs if len(refs[k][1]) >= 3]
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=50))
    matcher.add([refs[k][1] for k in names])
    matcher.train()
    local = cv2.BFMatcher()
    results = []
    checked = set()
    for k in names:
        xy, desc = refs[k]
        counts = Counter()
        for group in matcher.knnMatch(desc,k=8):
            counts.update(set(names[m.imgIdx] for m in group if names[m.imgIdx] != k and m.distance < 245))
        for other, votes in counts.most_common(14):
            pair = tuple(sorted((k,other)))
            if votes < 8 or pair in checked:
                continue
            checked.add(pair)
            pt, d = refs[other]
            good = [a for a,b in local.knnMatch(desc,d,k=2) if a.distance < .76*b.distance]
            if len(good) < 9:
                continue
            a = np.float32([xy[m.queryIdx] for m in good])
            b = np.float32([pt[m.trainIdx] for m in good])
            _, mask = cv2.findHomography(a,b,cv2.RANSAC,3.2)
            if mask is None:
                continue
            keep = mask.ravel().astype(bool)
            if sum(keep) >= 8 and min(cv2.contourArea(cv2.convexHull(v[keep])) for v in (a,b)) >= 600:
                results.append(dict(a=k,b=other,inliers=int(sum(keep)),fraction=round(keep.mean(),3)))
    args.output.write_text(json.dumps(results,indent=1))
    print("images",len(names),"candidate pairs",len(checked),"geometric matches",len(results))


if __name__ == "__main__":
    main()
