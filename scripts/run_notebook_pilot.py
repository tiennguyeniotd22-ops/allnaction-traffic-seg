"""Run the team's frozen notebook on a prepared ZIP from the VM workdir."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from notebook_identity import NOTEBOOK_TEAMS, TEAM_ID, verified_team
from train_ablation import CHANGES, configure_train_args


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
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Copy the saved EMA checkpoint every N completed epochs for offline scoring")
    parser.add_argument("--experiment", choices=CHANGES, default="control")
    parser.add_argument("--notebook", type=Path, default=Path("train_and_export.ipynb"))
    parser.add_argument("--warm-start", type=Path, help="Verified Team 21 run used for warm-polish initialization")
    args = parser.parse_args()
    if args.epochs < 1 or (args.patience is not None and args.patience < 1) or args.snapshot_every < 0:
        parser.error("Invalid epochs, patience, or snapshot interval")
    if (args.warm_start is not None) != (args.experiment == "warm-polish"):
        parser.error("--warm-start and the warm-polish recipe must be used together")
    run = args.run.resolve()
    dataset = args.dataset.resolve()
    notebook_sha = digest(args.notebook)
    weights_sha = digest(Path("yolo26n-seg.pt"))
    teacher_sha = None
    if args.experiment == "distill-x":
        teacher_sha = digest(Path(CHANGES["distill-x"]["distill_model"][1]))
        if teacher_sha != "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3":
            raise ValueError("Unexpected teacher checkpoint")
    cells = json.loads(args.notebook.read_text(encoding="utf-8"))["cells"]
    if NOTEBOOK_TEAMS.get(notebook_sha) != TEAM_ID:
        raise ValueError(f"Use the verified team-{TEAM_ID} training notebook")
    if weights_sha != "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f":
        raise ValueError("Initial weights differ from round 3")
    warm = None
    if args.warm_start is not None:
        parent = args.warm_start.resolve()
        origin = json.loads((parent / "run_report.json").read_bytes())
        info = json.loads((parent / "resources.json").read_bytes())
        assert verified_team(origin, info["notebook_sha256"]) == TEAM_ID and info["notebook_sha256"] == notebook_sha
        start_pt = parent / "runs/train/weights/best.pt"
        weights_sha = digest(start_pt)
        assert origin["train"]["best_sha256"] == weights_sha and origin["model"]["architecture"] == "yolo26n-seg"
        replay = json.loads((parent.parent / ("replay-cpu-" + parent.name) / "replay.json").read_bytes())
        assert replay["zip_sha256"] == digest(parent / "submission/submission.zip") == origin["submission"]["zip_sha256"]
        warm = dict(run=parent.name, weights_sha256=weights_sha, tensors_sha256=origin["train"]["best_tensors_sha256"],
                    source_zip_sha256=replay["zip_sha256"], source_dataset_sha256=info["dataset_sha256"])
    ns = {"__name__": "__main__"}

    def retain_snapshot(trainer):
        epoch = trainer.epoch + 1
        if epoch % args.snapshot_every == 0:
            target = run / "snapshots" / f"epoch-{epoch:03}.pt"
            target.parent.mkdir(exist_ok=True)
            temp = target.with_suffix(".partial")
            shutil.copyfile(trainer.last, temp)
            temp.replace(target)

    def cell(i):
        source = "".join(cells[i]["source"])
        if i == 15 and args.experiment != "control":
            marker = "model.train(**TRAIN_ARGS)"
            assert source.count(marker) == 1
            ns["configure_experiment"] = lambda cfg: configure_train_args(cfg, args.experiment, run)
            source = source.replace(marker, "configure_experiment(TRAIN_ARGS)\n" + marker)
        if i == 15 and args.snapshot_every:
            marker = "model.train(**TRAIN_ARGS)"
            assert source.count(marker) == 1
            ns["retain_snapshot"] = retain_snapshot
            source = source.replace(marker, 'model.add_callback("on_model_save", retain_snapshot)\n' + marker)
        if i == 15 and args.experiment in ("o2m-constant", "warm-polish"):
            from train_ablation import hold_export_branch_weight
            marker = "model.train(**TRAIN_ARGS)"
            assert source.count(marker) == 1
            ns["hold_export_branch_weight"] = hold_export_branch_weight
            source = source.replace(marker, 'model.add_callback("on_train_epoch_start", hold_export_branch_weight)\n' + marker)
        if i == 15 and args.experiment == "head-transfer":
            from team21_head_transfer import TrafficTransferTrainer
            marker = "model.train(**TRAIN_ARGS)"
            assert source.count(marker) == 1
            ns["TrafficTransferTrainer"] = TrafficTransferTrainer
            source = source.replace(marker, "model.train(trainer=TrafficTransferTrainer, **TRAIN_ARGS)")
        exec(compile(source, f"notebook-cell-{i}", "exec"), ns)

    for i in (1, 2, 3, 4, 5, 7):
        cell(i)
    assert ns["TEAM_ID"] == ns["PARAMS"]["TEAM_ID"] == TEAM_ID
    run.mkdir(parents=True, exist_ok=False)
    settings = dict(EPOCHS=args.epochs, BATCH=8, PATIENCE=args.patience or ns["PATIENCE"])
    ns.update(DATASET_PATH=str(dataset), WORK_DIR=str(run), DRIVE_OUTPUT_DIR="",
              INSTALL_PACKAGES=False, **settings)
    ns["PARAMS"].update(settings)
    for i in (9, 11, 13):
        cell(i)
    assert ns["SEED"] == TEAM_ID
    if warm:
        assert origin["classes"] == list(ns["CLASSES"]) and ns["weights_fingerprint"](start_pt) == warm["tensors_sha256"]
        ns["MODEL_WEIGHTS"] = str(start_pt)
        ns["PARAMS"]["MODEL_WEIGHTS"] = str(start_pt)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    cell(15)
    assert ns["TRAIN_INFO"]["train_args"]["seed"] == TEAM_ID
    resources = dict(revision=args.revision, experiment=args.experiment, team_id=TEAM_ID, dataset_sha256=digest(dataset),
                     initial_weights_sha256=weights_sha, notebook_sha256=notebook_sha,
                     gpu=torch.cuda.get_device_name(), fit_and_native_val_seconds=time.monotonic()-start,
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    if warm:
        resources["warm_start"] = warm
    if teacher_sha:
        resources["distillation_teacher_sha256"] = teacher_sha
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
