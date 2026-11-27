"""Run the team's frozen notebook on a prepared ZIP from the VM workdir."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--notebook", type=Path, default=Path("train_and_export.ipynb"))
    args = parser.parse_args()
    if args.epochs < 1 or (args.patience is not None and args.patience < 1):
        parser.error("epochs and patience must be positive")
    run = args.run.resolve()
    dataset = args.dataset.resolve()
    notebook_sha = digest(args.notebook)
    weights_sha = digest(Path("yolo26n-seg.pt"))
    cells = json.loads(args.notebook.read_text(encoding="utf-8"))["cells"]
    if notebook_sha != "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd":
        raise ValueError("Notebook differs from the frozen round-3 notebook")
    if weights_sha != "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f":
        raise ValueError("Initial weights differ from round 3")
    ns = {"__name__": "__main__"}

    def cell(i):
        exec(compile("".join(cells[i]["source"]), f"notebook-cell-{i}", "exec"), ns)

    for i in (1, 2, 3, 4, 5, 7):
        cell(i)
    run.mkdir(parents=True, exist_ok=False)
    settings = dict(EPOCHS=args.epochs, BATCH=8, PATIENCE=args.patience or ns["PATIENCE"])
    ns.update(DATASET_PATH=str(dataset), WORK_DIR=str(run), DRIVE_OUTPUT_DIR="",
              INSTALL_PACKAGES=False, **settings)
    ns["PARAMS"].update(settings)
    for i in (9, 11, 13):
        cell(i)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    cell(15)
    resources = dict(revision=args.revision, dataset_sha256=digest(dataset),
                     initial_weights_sha256=weights_sha, notebook_sha256=notebook_sha,
                     gpu=torch.cuda.get_device_name(), fit_and_native_val_seconds=time.monotonic()-start,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    (run/"resources.json").write_text(json.dumps(resources, indent=2))
    (run/"train_info.json").write_text(json.dumps(ns["TRAIN_INFO"], default=str, indent=2))
    for i in (17, 19, 21):
        cell(i)
    for key in ("PRED_PATH", "MANIFEST_PATH"):
        src = Path(ns[key])
        if src != run/src.name:
            shutil.copyfile(src, run/src.name)
    print("PILOT_COMPLETE", ns["EVAL_SUMMARY"], flush=True)


if __name__ == "__main__":
    main()
