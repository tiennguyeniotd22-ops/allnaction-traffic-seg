"""Exit successfully if an exact eligible ZIP has a verified val140 score >= 30."""
import hashlib
import json
from pathlib import Path

from notebook_identity import TEAM_ID, verified_team


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    root = Path("overnight")
    val_sha = digest(Path("audit_phase12/validation-manifest.json"))
    assert val_sha == "23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013"
    for path in sorted(root.glob("replay-cpu-*/replay.json")):
        replay = read(path)
        if replay["score"]["map50_95"] < .30 or replay.get("validation_sha256") != val_sha:
            continue
        run = root / path.parent.name.removeprefix("replay-cpu-")
        report = read(run / "run_report.json")
        if report.get("team_id") != TEAM_ID:
            continue
        info = read(run / "resources.json")
        assert verified_team(report, info["notebook_sha256"]) == TEAM_ID
        assert replay["notebook_sha256"] == info["notebook_sha256"] == digest(Path("train_team21.ipynb"))
        assert replay["dataset_sha256"] == info["dataset_sha256"]
        assert replay["prepared_sha256"] == report["dataset"]["sha256"]
        assert replay["score"]["n_images"] == 140 and sum(replay["score"]["n_gt_per_class"].values()) == 1588
        assert replay["zip_sha256"] == report["submission"]["zip_sha256"] == digest(run / "submission/submission.zip")
        print(run.name, replay["score"]["map50_95"], replay["zip_sha256"], flush=True)
        return 0
    return 3  # Distinguish an unmet threshold from an exception while checking the evidence.


if __name__ == "__main__":
    raise SystemExit(main())
