"""Describe a submitted model's val140 errors at confidence .25 using the frozen scorer."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pycocotools import mask as mu
from notebook_identity import verified_team


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    for arg in ("run", "notebook", "validation", "out"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    manifest = {r["file"]: r for r in read(a.validation)}
    report = read(a.run / "run_report.json")
    verified_team(report, sha(a.notebook))
    assert sha(a.run / "submission/submission.zip") == report["submission"]["zip_sha256"]
    mapping = read(a.run / "dataset/source_map.json")["files"]
    images = sorted((a.run / "dataset/images/val").glob("*.jpg"))
    originals = [Path(mapping["val/" + im.stem]).name for im in images]
    assert len(images) == len(manifest) == 140 and set(originals) == manifest.keys()
    for im, orig in zip(images, originals):
        assert sha(im) == manifest[orig]["image_sha256"]
        assert sha(a.run / "raw/labels/val" / (Path(orig).stem + ".txt")) == manifest[orig]["labels_sha256"]
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(read(a.notebook)["cells"][1:5], 1):
        exec(compile("".join(cell["source"]), f"notebook-{i}", "exec"), ns)
    prediction_file = a.run / "predictions_val.json"
    with TemporaryDirectory() as tmp:
        checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], Path(tmp))
        sc = checker.score
        params = sc.ScoringParams.from_config(checker.config)
        pred = sc._as_predictions(read(prediction_file), checker.config)
        examples = sc.build_images(pred, a.run / "dataset/labels/val", [im.stem for im in images], params)
        groups = sc.evaluate_images(examples, params)
        score = sc.accumulate(groups, np.ones(len(images), dtype=bool), params)
        assert abs(score["map50_95"] - report["val_estimate"]["map50_95"]) < 1e-9
        names = report["classes"]
        official = {}
        for c, name in enumerate(names):
            gt = sum(g.n_gt for g in groups[c])
            row = {"gt": gt}
            for t in (.5, .75, .9):
                i = list(np.round(params.thresholds, 2)).index(t)
                tp = sum(int(np.sum(g.dtm[i] & ~g.dtig[i] & (g.scores >= .25))) for g in groups[c])
                fp = sum(int(np.sum(~g.dtm[i] & ~g.dtig[i] & (g.scores >= .25))) for g in groups[c])
                row[str(t)] = dict(tp=tp, fp=fp, fn=gt-tp)
            official[name] = row

        causes = defaultdict(Counter)
        cross = Counter()
        for im in examples:
            keep = np.flatnonzero(np.asarray(im.gt_area) >= params.min_area)
            dc, conf = np.asarray(im.dt_cls), np.asarray(im.dt_score)
            targets = [im.gt_rles[k] for k in keep]
            if not targets:
                continue
            if len(dc):
                overlap = mu.iou(im.dt_rles, targets, [0]*len(targets))
                boxes = mu.iou(mu.toBbox(im.dt_rles), mu.toBbox(targets), [0]*len(targets))
            else:
                overlap = boxes = np.empty((0, len(keep)))
            for j, k in enumerate(keep):
                cls = im.gt_cls[k]
                iou, box = overlap[:, j], boxes[:, j]
                same, high = dc == cls, conf >= .25
                if np.any(same & high & (iou >= .5)):
                    tag = "correct_class_mask_at_conf25"
                elif np.any(~same & high & (iou >= .5)):
                    tag = "other_class_mask_at_conf25"
                    best = int(np.argmax(np.where(~same & high, iou, -1)))
                    cross[f"{names[cls]} -> {names[dc[best]]}"] += 1
                elif np.any(same & ~high & (iou >= .5)):
                    tag = "correct_mask_below_conf25"
                elif np.any(same & high & (box >= .5)):
                    tag = "same_class_box_matches_mask_below_50"
                else:
                    tag = "no_above_match"
                causes[names[cls]][tag] += 1
        assert all(sum(causes[name].values()) == official[name]["gt"] for name in names)
        conditions = {}
        keys = [manifest[n]["condition"] for n in originals]
        for key in sorted(set(keys)):
            select = np.array([k == key for k in keys])
            s = sc.accumulate(groups, select, params)
            conditions[key] = dict(images=int(select.sum()), map50_95=s["map50_95"], gt=s["n_gt_per_class"])
    result = dict(run=a.run.name, zip_sha256=report["submission"]["zip_sha256"],
                  val_manifest_sha256=sha(a.validation), predictions_sha256=sha(prediction_file),
                  official_score=score["map50_95"], confidence=.25, by_class=official, by_condition=conditions,
                  gt_overlap_diagnostics=causes, class_confusions=cross,
                  note="Operating counts use official one-to-one matching. The separate overlap diagnosis checks "
                       "each valid GT independently; one detection can overlap multiple GT. Condition mAP excludes "
                       "classes with no GT in that subset.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=1) + "\n")
    temp.replace(a.out)
    print(a.run.name, score["map50_95"], sum(r["gt"] for r in official.values()), "valid GT", flush=True)


if __name__ == "__main__":
    main()
