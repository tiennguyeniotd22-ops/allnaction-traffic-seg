"""Find train masks to inspect against the original images. Never write model proposals as labels."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu

from overnight_error_profile import read, sha


def show_case(image, gt, prediction, row, dest):
    w, h = image.size
    x, y, bw, bh = row["bbox"]
    pad = max(26, int(max(bw, bh) * .45))
    roi = (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad))
    canvas = np.array(image)
    for mask, color in [(gt, (0, 250, 255)), (prediction, (255, 34, 91))]:
        if mask is not None:
            binary = mu.decode(mask)
            assert binary.shape == (h, w)
            rings, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, rings, -1, color, 2)
    patch = image.crop(roi)
    factor = min(215 / patch.width, 188 / patch.height)
    size = tuple(max(1, round(v * factor)) for v in patch.size)
    tile = Image.new("RGB", (440, 228), "white")
    tile.paste(patch.resize(size), (0, 0))
    tile.paste(Image.fromarray(canvas).crop(roi).resize(size), (220, 0))
    title = f"{row['index']} {row['source']} | {row['name']} {row['confidence']:.2f}"
    ImageDraw.Draw(tile).text((2, 190), title + f"\n{row['reason']}, best GT IoU {row['max_iou']:.2f}", fill="black")
    tile.save(dest / f"{row['index']:03}.jpg", quality=94)
    return tile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "data_run", "notebook", "validation", "out"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    model = a.run / "submission/submission.zip"
    model_sha = sha(model)
    assert model_sha == read(a.run / "run_report.json")["submission"]["zip_sha256"]
    assert sha(a.notebook) == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    assert sha(a.validation) == "23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013"
    val_shas = {r["image_sha256"] for r in read(a.validation)}
    mapping = read(a.data_run / "dataset/source_map.json")["files"]
    images, sources = [], []
    for image in sorted((a.data_run / "dataset/images/train").glob("*.jpg")):
        source = Path(mapping["train/" + image.stem])
        if not re.fullmatch(r"(?:bdd_.*|vn_sign_.*|zz_drive_\d{12})\.jpg", source.name):
            continue
        assert source.parent.as_posix() == "images/train" and sha(image) == sha(a.data_run / "raw" / source)
        assert sha(image) not in val_shas
        images.append(image)
        sources.append(source.name)
    a.out.mkdir(parents=True, exist_ok=False)
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(read(a.notebook)["cells"][1:5], 1):
        exec(compile("".join(cell["source"]), f"official-cell-{i}", "exec"), ns)
    checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], a.out / "_checker")
    pred_path, _ = ns["run_platform_inference"](model, images, a.out, checker)
    sc = checker.score
    pred = sc._as_predictions(read(pred_path), checker.config)
    examples = sc.build_images(pred, a.data_run / "dataset/labels/train", [im.stem for im in images],
                               sc.ScoringParams.from_config(checker.config))
    cases = []
    for index, im in enumerate(examples):
        if not len(im.dt_rles):
            continue
        matrix = mu.iou(im.dt_rles, im.gt_rles, [0] * len(im.gt_rles)) if im.gt_rles else np.empty((len(im.dt_rles), 0))
        for j, (cls, conf) in enumerate(zip(im.dt_cls, im.dt_score, strict=True)):
            if conf < .35:
                continue
            best = int(np.argmax(matrix[j])) if matrix.shape[1] else None
            overlap = float(matrix[j, best]) if best is not None else 0
            compatible = [k for k, c in enumerate(im.gt_cls) if c == cls]
            same_iou = max((matrix[j, k] for k in compatible), default=0)
            if same_iou >= .4:
                continue
            if overlap < .12:
                reason = "possible missing mask"
            elif best is not None and im.gt_cls[best] != cls and overlap >= .55:
                reason = "class disagreement"
            else:
                continue
            cases.append((float(conf), index, j, best, int(cls), reason, overlap))
    tiles = a.out / "tiles"
    tiles.mkdir()
    selected, seen = [], set()
    counts = Counter()
    for conf, index, j, best, cls, reason, overlap in sorted(cases, key=lambda row: -row[0]):
        if (index, cls, reason) in seen or counts[cls, reason] >= 5:
            continue
        seen.add((index, cls, reason))
        counts[cls, reason] += 1
        im = examples[index]
        record = dict(index=len(selected), source=sources[index], class_id=cls, name=ns["CLASSES"][cls],
                      confidence=conf, reason=reason, max_iou=overlap, bbox=mu.toBbox(im.dt_rles[j]).astype(int).tolist(),
                      closest_gt_class=int(im.gt_cls[best]) if best is not None else None)
        with Image.open(images[index]) as image:
            tile = show_case(image.convert("RGB"), im.gt_rles[best] if best is not None and overlap else None,
                             im.dt_rles[j], record, tiles)
        if len(selected) % 9 == 0:
            sheet = Image.new("RGB", (1320, 684), "white")
        sheet.paste(tile, (len(selected) % 3 * 440, len(selected) % 9 // 3 * 228))
        selected.append(record)
        if len(selected) % 9 == 0:
            sheet.save(a.out / f"sheet-{(len(selected)-1)//9:02}.jpg", quality=93)
    if len(selected) % 9:
        sheet.save(a.out / f"sheet-{(len(selected)-1)//9:02}.jpg", quality=93)
    record = dict(zip_sha256=model_sha, training_dataset_sha256=read(a.data_run / "resources.json")["dataset_sha256"],
                  source_images=len(images), candidates=len(cases), selected=selected, prediction_sha256=sha(pred_path))
    (a.out / "review.json").write_text(json.dumps(record, indent=1) + "\n")
    print("Train images", len(images), "hypotheses for visual review", len(selected), flush=True)


if __name__ == "__main__":
    main()
