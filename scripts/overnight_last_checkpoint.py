"""Evaluate last.pt from a finished fit with the original competition export and decoder."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from ultralytics import YOLO


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ("parent", "out", "notebook", "validation"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    parent = a.parent.resolve()
    out = a.out.resolve()
    report = read(parent / "run_report.json")
    weights = parent / "runs/train/weights"
    last = weights / "last.pt"
    assert sha(weights / "best.pt") == report["train"]["best_sha256"]
    resources = read(parent / "resources.json")
    assert sha(a.notebook) == resources["notebook_sha256"] == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    assert version("ultralytics") == report["environment"]["packages"]["ultralytics"] == "8.4.214"
    assert report["seed"] == report["team_id"] == 26 and report["model"]["architecture"] == "yolo26n-seg"
    if out.exists():
        prior = read(out / "checkpoint-selection.json")
        assert prior["last_sha256"] == sha(last) and prior["parent_submission_sha256"] == report["submission"]["zip_sha256"]
        if prior["identical_tensors"] is False:
            assert sha(out / "submission/submission.zip") == read(out / "run_report.json")["submission"]["zip_sha256"]
        print("Already evaluated", out.name, flush=True)
        return
    cells = read(a.notebook)["cells"]
    ns = {"__name__": "__main__"}
    for i in (1, 2, 3, 4, 5):
        exec(compile("".join(cells[i]["source"]), f"notebook-{i}", "exec"), ns)
    before = ns["weights_fingerprint"](weights / "best.pt")
    assert before == report["train"]["best_tensors_sha256"]
    after = ns["weights_fingerprint"](last)
    choice = dict(parent=str(parent), parent_submission_sha256=report["submission"]["zip_sha256"],
                  parent_weights_sha256=sha(weights / "best.pt"), parent_tensors_sha256=before,
                  last_sha256=sha(last), last_tensors_sha256=after, identical_tensors=before == after)
    # The original fit is immutable. Work in a new, atomically published directory.
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as root:
        run = Path(root) / "work"
        run.mkdir()
        write(run / "checkpoint-selection.json", choice)
        if before == after:
            run.rename(out)
            print("Same tensors; parent ZIP already evaluated", parent, flush=True)
            return
        for folder in ("dataset", "raw"):
            (run / folder).symlink_to(parent / folder, target_is_directory=True)
        artifacts = run / "runs/train"
        local_pt = artifacts / "weights/best.pt"
        local_pt.parent.mkdir(parents=True)
        shutil.copyfile(last, local_pt)
        for name in ("args.yaml", "results.csv"):
            shutil.copyfile(parent / "runs/train" / name, artifacts / name)
        checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], run / "_checker")
        files = sorted((run / "dataset/images/val").glob("*.jpg"))
        source_files = read(run / "dataset/source_map.json")["files"]
        manifest = {r["file"]: r for r in read(a.validation)}
        assert len(files) == len(manifest) == 140
        assert {Path(source_files["val/" + f.stem]).name for f in files} == manifest.keys()
        for f in files:
            original = Path(source_files["val/" + f.stem]).name
            assert sha(f) == manifest[original]["image_sha256"]
            assert sha(run / "raw/labels/val" / (Path(original).stem + ".txt")) == manifest[original]["labels_sha256"]
        assert ns["dataset_fingerprint"](run / "dataset") == report["dataset"]["sha256"]
        submission = ns["make_submission"](local_pt, run, YOLO, checker)
        pred, perf = ns["run_platform_inference"](submission["zip"], files, run, checker)
        labels = run / "dataset/labels/val"
        result = ns["score_predictions"](pred, labels, files, checker).to_dict()
        errors, confusion = ns["analyze_errors"](pred, labels, checker, .25)
        scores = [d["s"] for im in read(pred)["images"] for d in im["instances"]]
        write(run / "diagnostics.json", dict(approximate_error_conf=.25, iou=.5, errors=errors,
              confusion=[(int(x), int(y), int(n)) for (x, y), n in confusion.items()],
              confidence_quantiles=np.quantile(scores, [.1,.5,.9]).tolist() if scores else [], inference=perf))
        trainer = dict(report["train"], best_weights=str(out / "runs/train/weights/best.pt"),
                       best_sha256=sha(local_pt), best_tensors_sha256=after, source_checkpoint=str(last))
        eval_result = dict(map50_95=result["map50_95"], score_x100=result["score"], map50=result["map50"],
                           per_class_ap=result["per_class_ap"], n_images=result["n_images"],
                           latency_ms_p50=perf["latency_ms_p50"], note="Official ONNX self-check on the frozen team validation split.")
        # Preserve the verified parent fit metadata. The official manifest builder creates
        # the new export fields from actual output, then the inherited dataset stats are checked.
        data = report["dataset"]
        audit = ns["analyze_dataset"](run / "dataset", report["classes"], checker.config.min_gt_area_px, 640)
        new = ns["build_manifest"](26, 26, report["params"], data, audit, trainer, submission, eval_result,
                                   {"installed":report["environment"]["packages"]}, report["classes"])
        assert new["dataset"] == data
        assert new["model"]["architecture"] == "yolo26n-seg" and new["model"]["nc"] == 11
        new["checkpoint_selection"] = choice
        ns["write_manifest"](new, run / "run_report.json")
        write(run / "train_info.json", trainer)
        write(run / "resources.json", dict(resources, reuses_fit_from=str(parent), additional_fit_seconds=0))
        run.rename(out)
        print("Last-epoch official val140", parent.name, result["score"], flush=True)


if __name__ == "__main__":
    main()
