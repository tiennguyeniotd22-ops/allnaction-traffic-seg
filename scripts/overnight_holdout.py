"""Score class 3 with the official ONNX pipeline on the pre-inference frozen pack."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "holdout", "manifest", "notebook", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert digest(args.holdout.read_bytes()) == manifest["holdout_sha256"]
    assert digest(args.notebook.read_bytes()) == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    args.out.mkdir(parents=True, exist_ok=True)
    records = [r for r in manifest["records"] if r["role"] == "holdout"]
    images, labels = args.out / "images", args.out / "labels"
    for directory in (images, labels):
        directory.mkdir(exist_ok=True)
    with ZipFile(args.holdout) as archive:
        assert len(archive.namelist()) == 2 * len(records)
        for row in records:
            for directory, suffix, field in ((images, ".jpg", "image_sha256"), (labels, ".txt", "label_sha256")):
                name = row["stem"] + suffix
                data = archive.read(f"{directory.name}/val/{name}")
                assert digest(data) == row[field]
                (directory / name).write_bytes(data)
    namespace = {"__name__": "__main__"}
    cells = json.loads(args.notebook.read_text())["cells"]
    for index in (1, 2, 3, 4):
        exec(compile("".join(cells[index]["source"]), f"notebook-{index}", "exec"), namespace)
    checker = namespace["load_selfcheck"](namespace["SELFCHECK_BUNDLE"], args.out / "_checker")
    files = sorted(images.glob("*.jpg"))
    assert len(files) == len(records) == len(list(labels.glob("*.txt")))
    path = args.run / "submission/submission.zip"
    predictions, _ = namespace["run_platform_inference"](path, files, args.out, checker)
    result = namespace["score_predictions"](predictions, labels, files, checker).to_dict()
    assert set(result["per_class_ap"]) == {"Other vehicle"}
    summary = {key: result[key] for key in ("score", "map50", "map50_95", "per_class_ap", "n_images", "n_gt_per_class", "n_gt_ignored")}
    summary.update(zip_sha256=digest(path.read_bytes()), holdout_sha256=manifest["holdout_sha256"],
                   gt_instances=sum(len(row["objects"]) for row in records), run=str(args.run))
    (args.out / "score.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
