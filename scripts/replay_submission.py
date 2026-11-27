"""Rebuild the notebook dataset and score its submission on CPU without running a fit."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("notebook", "dataset", "submission", "report", "resources", "validation", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    resources = json.loads(args.resources.read_bytes())
    manifest = json.loads(args.validation.read_bytes())
    assert sha(args.notebook) == resources["notebook_sha256"] == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    assert sha(args.dataset) == resources["dataset_sha256"]
    assert sha(args.submission) == report["submission"]["zip_sha256"]
    assert sha(args.validation) == "23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013"
    assert report["team_id"] == report["seed"] == 26 and len(manifest) == 140
    with ZipFile(args.submission) as archive:
        assert archive.namelist() == ["model.onnx"]
        assert hashlib.sha256(archive.read("model.onnx")).hexdigest() == report["model"]["onnx_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    cells = json.loads(args.notebook.read_bytes())["cells"]
    ns = {"__name__": "__main__"}
    for i in range(1, 5):
        exec(compile("".join(cells[i]["source"]), f"notebook-{i}", "exec"), ns)
    with TemporaryDirectory(dir=args.out) as temp:
        work = Path(temp).resolve()
        raw = ns["locate_dataset"](args.dataset.resolve(), work)
        for row in manifest:
            name = row["file"]
            assert sha(raw / "images/val" / name) == row["image_sha256"]
            assert sha(raw / "labels/val" / (Path(name).stem + ".txt")) == row["labels_sha256"]
        prepared = work / "dataset"
        ns["prepare_dataset"](raw, prepared, report["classes"], report["seed"])
        assert ns["dataset_fingerprint"](prepared) == report["dataset"]["sha256"]
        checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], work / "checker")
        images = sorted((prepared / "images/val").glob("*.jpg"))
        files = json.loads((prepared / "source_map.json").read_bytes())["files"]
        assert len(images) == len(manifest)
        assert {Path(files["val/" + im.stem]).name for im in images} == {r["file"] for r in manifest}
        predictions, perf = ns["run_platform_inference"](args.submission, images, work, checker)
        score = ns["score_predictions"](predictions, prepared / "labels/val", images, checker).to_dict()
        expected = report["val_estimate"]
        for name in ("map50_95", "map50"):
            assert abs(score[name] - expected[name]) <= 1e-8, (name, score[name], expected[name])
        assert score["per_class_ap"].keys() == expected["per_class_ap"].keys()
        assert all(abs(v - expected["per_class_ap"][c]) <= 1e-8 for c, v in score["per_class_ap"].items())
        assert sum(score["n_gt_per_class"].values()) == 1588
        result = dict(score=score, cpu=perf, zip_sha256=sha(args.submission), dataset_sha256=sha(args.dataset),
                      prepared_sha256=report["dataset"]["sha256"], notebook_sha256=sha(args.notebook),
                      validation_sha256=sha(args.validation), predictions_sha256=sha(predictions))
        predictions.replace(args.out / "predictions_val.json")
    (args.out / "replay.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(score["score"], result["zip_sha256"], flush=True)


if __name__ == "__main__":
    main()
