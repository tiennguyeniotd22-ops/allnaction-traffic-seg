"""Compare the completed E7 fit and its retained checkpoints, then replay the winning ZIP."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from notebook_identity import TEAM_ID, verified_team
from overnight_final_queue import PYTHON, ROOT, digest, read, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--version", default="e3-triage-v1")
    parser.add_argument("--notebook", type=Path, default=Path("train_team21.ipynb"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Use CPU for checkpoint export and validation")
    notebook = args.notebook
    validation = Path("audit_phase12/validation-manifest.json")
    version = args.version
    parent = ROOT / args.fit
    rf = read(parent / "resources.json")
    dataset_sha = digest(ROOT / version / "dataset.zip")
    assert rf["dataset_sha256"] == dataset_sha and rf["notebook_sha256"] == digest(notebook)
    assert verified_team(read(parent / "run_report.json"), digest(notebook)) == TEAM_ID
    candidates = {}
    code = Path(__file__).resolve().parent

    def run(script, *params):
        subprocess.run([PYTHON, code / script, *map(str, params)], check=True)

    def validate(name, reference):
        dest, base = ROOT / name, ROOT / reference
        scores = ROOT / "scores" / name
        scores.mkdir(parents=True, exist_ok=True)
        assert verified_team(read(dest / "run_report.json"), digest(notebook)) == TEAM_ID
        if not (scores / "held/score.json").exists():
            run("overnight_holdout.py", "--run", dest, "--holdout", ROOT / "e2-v1/holdout.zip",
                "--manifest", ROOT / "e2-v1/manifest.json", "--notebook", notebook, "--out", scores / "held")
        if not (scores / "comparison.json").exists():
            cmd = [PYTHON, code / "compare_pilot.py", "--baseline", base, "--pilot", dest, "--notebook", notebook,
                   "--manifest", validation, "--out", scores / "comparison.json", "--allow-schedule-change"]
            if read(base / "run_report.json")["seed"] == 26:
                cmd.append("--historical-team26")
            else:
                source_recipe = read(base / "resources.json")["experiment"]
                target_recipe = read(dest / "resources.json")["experiment"]
                if source_recipe != target_recipe:
                    cmd.extend(["--reference-experiment", source_recipe, "--experiment", target_recipe])
            ret = subprocess.run(cmd)
            if ret.returncode == 65:
                decoded = ROOT / "decoded" / name
                run("overnight_decode_parity.py", dest, notebook, validation, decoded)
                subprocess.run([*cmd, "--decoded-parity", decoded / "decoded-comparison.json"], check=True)
            else:
                ret.check_returncode()
        result = verified(name)
        assert result is not None
        return result

    candidates[args.fit] = validate(args.fit, args.reference)
    if read(ROOT / args.reference / "run_report.json")["seed"] == TEAM_ID:
        comparable = verified(args.reference)
        assert comparable is not None
        candidates[args.reference] = comparable

    def export(name, epoch=None):
        dest = ROOT / name
        options = [] if epoch is None else ["--epoch", epoch]
        run("overnight_last_checkpoint.py", parent, dest, notebook, validation, *options)
        choice = read(dest / "checkpoint-selection.json")
        if choice["identical_tensors"]:
            return candidates[args.fit]
        return read(dest / "run_report.json")["val_estimate"]["map50_95"]

    def verify_candidate(name, score):
        if score <= max(candidates.values()):
            return
        # An existing export is required. This invocation cannot start a new fit.
        assert (ROOT / name / "run_report.json").is_file()
        assert validate(name, args.fit) == score
        candidates[name] = score

    last = "last-" + args.fit
    trials = {last: export(last)}
    verify_candidate(last, trials[last])
    screens = []
    for pt in sorted((parent / "snapshots").glob("epoch-*.pt")):
        result = ROOT / "screens" / (args.fit + "-" + pt.stem + ".json")
        # The live screening process usually produced this file during training.
        if not result.is_file():
            run("overnight_screen_checkpoint.py", pt, ROOT / args.reference, notebook, validation, result)
        record = read(result)
        assert record["identity"]["weights_sha256"] == digest(pt)
        assert record["identity"]["validation_sha256"] == digest(validation)
        assert record["score"]["n_images"] == 140
        screens.append((record["score"]["map50_95"], int(pt.stem.split("-")[-1])))
    for screen_score, epoch in sorted(screens, reverse=True):
        name = f"{args.fit}-epoch-{epoch:03}"
        trials[name] = screen_score
        if screen_score <= max(candidates.values()):
            continue
        score = export(name, epoch)
        assert abs(score - screen_score) < 1e-8
        verify_candidate(name, score)
    best = max(candidates, key=candidates.get)
    folder = ROOT / best
    winner_sha = read(folder / "resources.json")["dataset_sha256"]
    if winner_sha != dataset_sha:
        matches = [p.parent for p in ROOT.glob("*/manifest.json")
                   if (p.parent / "dataset.zip").exists() and isinstance((m := read(p)), dict)
                   and m.get("dataset_sha256") == winner_sha]
        assert len(matches) == 1 and digest(matches[0] / "dataset.zip") == winner_sha
        version, dataset_sha = matches[0].name, winner_sha
    model = folder / "submission/submission.zip"
    replay = ROOT / ("replay-cpu-" + best)
    if not (replay / "replay.json").is_file():
        run("replay_submission.py", "--notebook", notebook, "--dataset", ROOT / version / "dataset.zip",
            "--submission", model, "--report", folder / "run_report.json", "--resources", folder / "resources.json",
            "--validation", validation, "--out", replay)
    result = read(replay / "replay.json")
    assert result["dataset_sha256"] == dataset_sha and result["validation_sha256"] == digest(validation)
    assert result["zip_sha256"] == digest(model) and abs(result["score"]["map50_95"] - candidates[best]) < 1e-8
    probe = ROOT / "probe-scores" / best
    run("overnight_score_probe.py", ROOT / "probe-frozen.zip", ROOT / "probe-lock.json", notebook, model, probe)
    probe_score = read(probe / "score.json")
    assert probe_score["zip_sha256"] == digest(model) and probe_score["n_images"] == 46
    profile = ROOT / "scores" / best / "error-profile.json"
    run("overnight_error_profile.py", folder, notebook, validation, profile)
    assert read(profile)["zip_sha256"] == digest(model)
    record = dict(winner=best, verified_scores=candidates, checkpoint_scores=trials, zip_sha256=digest(model),
                  replay_map=result["score"]["map50_95"], probe_map=probe_score["map50_95"])
    path = ROOT / (args.fit + "-result.json")
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(record, indent=1) + "\n")
    temp.replace(path)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
