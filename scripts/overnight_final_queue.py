"""Run the reviewed data pilots and choose one matched recipe for full training on L4."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime


ROOT = Path("overnight")
CODE = Path(os.getenv("PLANTCARE_TRAIN_SCRIPTS", ROOT / "ablation"))
PYTHON = ".venv/bin/python"
DATA = {
    "e3-tax-v2": "d9fabdc563c4de14d4cb8205dcd10e6ffcc849186cfe69c8fa1a20b11efc0dad",
    "e3-weather-v1": "98c08b696635b776991aa26d8f0eafbbf8781c45a6bfb4c25dcbc9bbac06dbb8",
    "e3-weatherqc-v1": "f125ebfd5846fe6c948dd6c1bb33ce70c63cfb4fd7a505e7453e709d8af70a5e",
    "e3-riderqc-v1": "90f7cd78bacc169d02e9d7bb828d83e7150f487e3f7f9b1fed3d28f39050b59c",
    "e3-bdd-v1": "5e056fb7fead9c24f78db2f7057d77ea57743fa5a4c15aad01b6857b31f5199e",
    "e3-vistas-v1": "3cf51625f61792c51914e5f3a661f25abe4d5380f69c1f8f4c3cc3d2329afc0d",
    "e3-vnrepair-v1": "03b99ac7680044dfb2ae2b1e22839d78fcfc20cf853e6c33c39b2b4d5e52daa2",
}
PILOTS = {
    "e3-tax20": "e3-tax-v2", "e4-repeat20": "e3-tax-v2", "e4-nocopy20": "e3-tax-v2",
    "e5-mask420": "e3-tax-v2", "e6-lowerlr20": "e3-tax-v2", "e6-linear20": "e3-tax-v2",
    "e6-sgd20": "e3-tax-v2", "e3-weather20": "e3-weather-v1",
    "e3-bdd20": "e3-bdd-v1", "e3-vistas20": "e3-vistas-v1",
    "e3-vnrepair20": "e3-vnrepair-v1",
    "e3-weatherqc20": "e3-weatherqc-v1",
    "e3-riderqc20": "e3-riderqc-v1",
}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def score_dir(name):
    return ROOT / "scores" / name


def verified(name):
    folder = ROOT / name
    out = score_dir(name)
    if not (out / "comparison.json").exists() or not (out / "held/score.json").exists():
        return None
    record = read(out / "comparison.json")
    assert record["zip_sha256"] == read(out / "held/score.json")["zip_sha256"] == digest(folder / "submission/submission.zip")
    assert record["validation_sha_verified"] == 140 and record["opsets"][""] == 17
    assert len(record["onnx_parity"]) == 6
    score = record["scores"]["pilot"]["all140"]["map50_95"]
    assert score == read(folder / "run_report.json")["val_estimate"]["map50_95"]
    return score


def train_and_score(name, version, reference, recipe, revision, epochs, patience, final=False):
    run = ROOT / name
    dataset = ROOT / version / "dataset.zip"
    assert digest(dataset) == DATA[version]
    if not (run / "run_report.json").exists():
        if run.exists():
            raise RuntimeError(f"Incomplete existing run at {run}")
        with (ROOT / (name + ".log")).open("a") as f:
            subprocess.run([PYTHON, CODE / "run_notebook_pilot.py", "--dataset", dataset, "--run", run,
                            "--revision", revision, "--epochs", str(epochs), "--patience", str(patience),
                            "--experiment", recipe, *(["--snapshot-every", "20"] if final else [])],
                           check=True, stdout=f, stderr=subprocess.STDOUT)
    resources = read(run / "resources.json")
    assert resources["dataset_sha256"] == DATA[version] and resources["experiment"] == recipe
    out = score_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (ROOT / (name + "score.log")).open("a") as f:
        if not (out / "held/score.json").exists():
            subprocess.run([PYTHON, ROOT / "overnight_holdout.py", "--run", run, "--holdout", ROOT / "e2-v1/holdout.zip",
                            "--manifest", ROOT / "e2-v1/manifest.json", "--notebook", "train_and_export.ipynb",
                            "--out", out / "held"], check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
        if not (out / "comparison.json").exists():
            same_recipe = read(ROOT / reference / "resources.json").get("experiment", "control") == recipe
            options = ["--allow-schedule-change"] if final else ["--experiment", "control" if same_recipe else recipe]
            compare = [PYTHON, CODE / "compare_pilot.py", "--baseline", ROOT / reference, "--pilot", run,
                            "--notebook", "train_and_export.ipynb", "--manifest", "audit_phase12/validation-manifest.json",
                            "--out", out / "comparison.json", *options]
            io = dict(env=env, stdout=f, stderr=subprocess.STDOUT)
            result = subprocess.run(compare, **io)
            if result.returncode == 65:
                decoded = ROOT / "decoded" / name
                subprocess.run([PYTHON, CODE / "overnight_decode_parity.py", run, "train_and_export.ipynb",
                                "audit_phase12/validation-manifest.json", decoded], check=True, **io)
                subprocess.run([*compare, "--decoded-parity", decoded / "decoded-comparison.json"], check=True, **io)
            else:
                result.check_returncode()
    print(name, "verified", verified(name), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", required=True, help="ISO timestamp with an explicit time zone")
    p.add_argument("--revision", required=True)
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        p.error("Deadline must include a UTC offset")
    for session in ("weathertrials", "parametertrials"):
        while subprocess.run(["tmux", "has-session", "-t", "=" + session], stderr=subprocess.DEVNULL).returncode == 0:
            time.sleep(10)
    for name, base, recipe, pilot_minutes in [
        ("e3-bdd20", "e3-weather20", "control", 16),
        ("e3-vistas20", "e3-bdd20", "control", 20),
        ("e6-lowerlr20", "e3-tax20", "lower-lr", 15),
        ("e6-linear20", "e3-tax20", "linear-lr", 15),
        ("e3-riderqc20", "e6-lowerlr20", "lower-lr", 16),
        ("e6-sgd20", "e3-tax20", "sgd", 15),
        ("e3-weatherqc20", "e3-weather20", "control", 16),
        ("e3-vnrepair20", "e3-vistas20", "control", 20),
    ]:
        if verified(name) is not None:
            continue
        remaining_pilot = 3 if (ROOT / name / "run_report.json").is_file() else pilot_minutes
        if deadline.timestamp() - time.time() < (60 + remaining_pilot) * 60:
            print("Reserving remaining time for E7", flush=True)
            break
        if verified(base) is None:
            raise RuntimeError(f"Expected verified matched data pilot {base}")
        try:
            train_and_score(name, PILOTS[name], base, recipe, args.revision, 20, 20)
        except subprocess.CalledProcessError as exc:
            # A scoring failure quarantines only this candidate; a fit failure needs diagnosis.
            if not (ROOT / name / "run_report.json").is_file():
                raise
            print(f"{name} excluded pending verification, exit {exc.returncode}: {exc.cmd}", flush=True)
    scores = {name: value for name in PILOTS if (value := verified(name)) is not None}
    best = max(scores, key=scores.get)
    recipe = read(ROOT / best / "resources.json").get("experiment", "control")
    assert read(ROOT / best / "resources.json")["dataset_sha256"] == DATA[PILOTS[best]]
    with (ROOT / best / "runs/train/results.csv").open() as f:
        times = [float(r["time"]) for r in csv.DictReader(f)]
    per_epoch = statistics.median(b - a for a, b in zip(times, times[1:]))
    # Reserve ten minutes for startup, export, CPU checks, and local handoff.
    epochs = min(100, int((deadline.timestamp() - time.time() - 600) / (1.10 * per_epoch)))
    if epochs < 25:
        raise RuntimeError(f"Insufficient time for E7 ({epochs} epochs)")
    decision = dict(pilot=best, pilot_scores=scores, metric="official ONNX val140 mask mAP50-95",
                    version=PILOTS[best], recipe=recipe, dataset_sha256=DATA[PILOTS[best]], epochs=epochs,
                    patience=25, median_epoch_seconds=per_epoch, deadline=args.deadline, revision=args.revision)
    decision["snapshot_interval_epochs"] = 20
    print(json.dumps(decision), flush=True)
    plan = ROOT / "e7-selection.json"
    if plan.exists() and read(plan) != decision:
        raise RuntimeError("Existing E7 selection differs")
    plan.write_text(json.dumps(decision, indent=2))
    train_and_score("e7-final", PILOTS[best], best, recipe, args.revision, epochs, 25, final=True)


if __name__ == "__main__":
    main()
