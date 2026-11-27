"""Score one atomically saved checkpoint, including during a fit, on the fixed val140."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from ultralytics import YOLO


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser()
    for name in ("weights", "reference", "notebook", "validation", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        p.error("Set CUDA_VISIBLE_DEVICES to an empty string for CPU screening")
    resource = read(a.reference / "resources.json")
    report = read(a.reference / "run_report.json")
    assert digest(a.notebook) == resource["notebook_sha256"] == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    assert version("ultralytics") == report["environment"]["packages"]["ultralytics"] == "8.4.214"
    assert report["seed"] == report["team_id"] == 26 and report["model"]["nc"] == 11
    identity = dict(weights_sha256=digest(a.weights), checkpoint=a.weights.as_posix(),
                    validation_sha256=digest(a.validation), reference_zip=report["submission"]["zip_sha256"],
                    notebook_sha256=resource["notebook_sha256"])
    if a.output.is_file():
        prior = read(a.output)
        assert prior["identity"] == identity and prior["score"]["n_images"] == 140
        print(a.weights.name, prior["score"]["score"], flush=True)
        return
    manifest = {row["file"]: row for row in read(a.validation)}
    source = read(a.reference / "dataset/source_map.json")["files"]
    images = sorted((a.reference / "dataset/images/val").glob("*.jpg"))
    original = [Path(source["val/" + f.stem]).name for f in images]
    assert len(images) == len(manifest) == 140 and set(original) == manifest.keys()
    for image, name in zip(images, original, strict=True):
        label = a.reference / "raw/labels/val" / (Path(name).stem + ".txt")
        assert digest(image) == manifest[name]["image_sha256"] and digest(label) == manifest[name]["labels_sha256"]
    cells = read(a.notebook)["cells"]
    ns = {"__name__": "__main__"}
    for i in range(1, 6):
        exec(compile("".join(cells[i]["source"]), f"notebook-{i}", "exec"), ns)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=a.output.parent) as temp:
        run = Path(temp)
        weights = run / "best.pt"
        shutil.copyfile(a.weights, weights)
        assert digest(weights) == identity["weights_sha256"]
        checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], run / "_checker")
        submission = ns["make_submission"](weights, run, YOLO, checker)
        pred, perf = ns["run_platform_inference"](submission["zip"], images, run, checker)
        score = ns["score_predictions"](pred, a.reference / "dataset/labels/val", images, checker).to_dict()
        assert score["n_images"] == 140 and sum(score["n_gt_per_class"].values()) == 1588
        result = dict(identity=identity, score=score, onnx_sha256=submission["model_sha256"],
                      tensors_sha256=ns["weights_fingerprint"](weights), cpu_inference=perf)
    temp = a.output.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=1) + "\n")
    temp.replace(a.output)
    print(a.weights.name, score["score"], flush=True)


if __name__ == "__main__":
    main()
