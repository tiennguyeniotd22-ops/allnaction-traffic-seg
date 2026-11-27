"""Score FP32 PyTorch and the submitted ONNX with the frozen official mask decoder."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as masks
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO
from notebook_identity import verified_team


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mask_matches(native, exported):
    counts = Counter()
    overlaps = []
    for left, right in zip(native["images"], exported["images"], strict=True):
        assert (left["file"], left["width"], left["height"]) == (right["file"], right["width"], right["height"])
        for cls in range(11):
            a, b = [[{"size": [im["height"], im["width"]], "counts": obj["rle"].encode("ascii")}
                     for obj in im["instances"] if obj["c"] == cls and obj["s"] >= .25] for im in (left, right)]
            counts.update(native=len(a), exported=len(b))
            if not a or not b:
                continue
            iou = masks.iou(a, b, [0] * len(b))
            rows, cols = linear_sum_assignment(iou, maximize=True)
            overlaps.extend(iou[rows, cols].tolist())
    counts["match_iou_ge_99"] = sum(v >= .99 for v in overlaps)
    counts["match_iou_eq_1"] = sum(v == 1 for v in overlaps)
    return dict(counts), np.quantile(overlaps, [0, .01, .5, 1]).tolist() if overlaps else []


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "notebook", "manifest", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    root, out = args.run, args.out
    report = read(root / "run_report.json")
    weights = root / "runs/train/weights/best.pt"
    submission = root / "submission/submission.zip"
    assert sha(weights) == report["train"]["best_sha256"]
    assert sha(submission) == report["submission"]["zip_sha256"]
    verified_team(report, sha(args.notebook))
    assert version("ultralytics") == "8.4.214"
    files = sorted((root / "dataset/images/val").glob("*.jpg"))
    manifest = {r["file"]: r for r in read(args.manifest)}
    source = read(root / "dataset/source_map.json")["files"]
    assert len(files) == len(manifest) == 140
    originals = [Path(source["val/" + f.stem]).name for f in files]
    assert set(originals) == manifest.keys()
    labels = root / "dataset/labels/val"
    for f, name in zip(files, originals):
        assert sha(f) == manifest[name]["image_sha256"]
        assert sha(root / "raw/labels/val" / (Path(name).stem + ".txt")) == manifest[name]["labels_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    ns = {"__name__": "__main__"}
    cells = read(args.notebook)["cells"]
    for i in (1, 2, 3, 4):
        exec(compile("".join(cells[i]["source"]), f"notebook-{i}", "exec"), ns)
    checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], out / "_checker")
    cfg, infer = checker.config, checker.infer
    params = checker.decode.DecodeParams.from_config(cfg)
    ort_dir = out / "ort"
    ort_dir.mkdir(exist_ok=True)
    exported, perf = ns["run_platform_inference"](submission, files, ort_dir, checker)
    ort_score = ns["score_predictions"](exported, labels, files, checker).to_dict()
    assert abs(ort_score["map50_95"] - report["val_estimate"]["map50_95"]) < 1e-9
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = False
    cv2.setNumThreads(1)
    net = YOLO(str(weights)).model.cpu().float().eval().fuse(verbose=False)
    head = net.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False
    native = []
    with torch.inference_mode():
        for path in files:
            x, meta = infer.preprocess(path, params.imgsz, params.pad_value)
            start = time.perf_counter()
            pred, proto = net(torch.from_numpy(x.copy()))
            after = time.perf_counter()
            assert pred.shape == (1, 47, 8400) and proto.shape == (1, 32, 160, 160)
            objects = checker.decode.decode_image(pred.numpy(), proto.numpy(), meta, params)
            native.append(infer.ImagePrediction(path.name, meta.orig_h, meta.orig_w, objects,
                                                (after - start) * 1000, (time.perf_counter() - after) * 1000))
    native_path = out / "native-predictions.json"
    checker.cli_infer.write_predictions(infer.InferenceResult(images=native, threads=2), native_path, params, report["classes"])
    native_score = ns["score_predictions"](native_path, labels, files, checker).to_dict()
    keys = ("map50", "map75", "map50_95", "per_class_ap", "n_detections", "n_gt_ignored", "n_images", "n_gt_per_class")
    assert native_score["n_gt_per_class"] == ort_score["n_gt_per_class"]
    counts, quantiles = mask_matches(read(native_path), read(exported))
    diff = {k: 100 * (ort_score[k] - native_score[k]) for k in ("map50", "map75", "map50_95")}
    per_class = {k: 100 * (v - native_score["per_class_ap"][k]) for k, v in ort_score["per_class_ap"].items()}
    # The secondary gate is application-level: preserve aggregate and per-class AP, and the
    # masks of confident instances. All deltas are percentage points on the 0–100 score.
    passed = (max(map(abs, diff.values())) <= .02 and max(map(abs, per_class.values())) <= .10
              and counts["match_iou_ge_99"] >= .995 * max(counts["native"], counts["exported"], 1))
    result = dict(run=root.name, zip_sha256=sha(submission), best_sha256=sha(weights),
                  validation_manifest_sha256=sha(args.manifest), notebook_sha256=sha(args.notebook),
                  native={k:native_score[k] for k in keys}, onnx={k:ort_score[k] for k in keys},
                  delta_points=diff, per_class_delta_points=per_class, high_confidence=counts,
                  high_confidence_matched_mask_iou_min_p01_median_max=quantiles,
                  onnx_latency_ms_p50=perf["latency_ms_p50"], passed=passed)
    temp = out / "decoded-comparison.partial"
    temp.write_text(json.dumps(result, indent=1) + "\n")
    temp.replace(out / "decoded-comparison.json")
    print(root.name, diff, counts, "passed", passed, flush=True)


if __name__ == "__main__":
    main()
