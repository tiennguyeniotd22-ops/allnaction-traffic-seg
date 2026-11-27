"""Fetch a finished VM run and verify all model hashes before recording scores."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tarfile
from zipfile import ZipFile


def digest(body):
    return hashlib.sha256(body).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out", type=Path)
    p.add_argument("scoreboard", type=Path)
    p.add_argument("--ssh", default="plantcare-cvat")
    args = p.parse_args()
    if not re.fullmatch("[a-z0-9-]+", args.run):
        p.error("Invalid run directory name")
    base = "overnight/" + args.run + "/"
    scores = "overnight/scores/" + args.run + "/"
    files = {base+s:Path(s).name for s in ("submission/submission.zip", "runs/train/weights/best.pt",
        "runs/train/args.yaml", "runs/train/results.csv", "run_report.json", "resources.json", "train_info.json")}
    files.update({scores+"comparison.json":"comparison.json", scores+"held/score.json":"held-score.json"})
    if "repeat" in args.run:
        files[base+"sampling.json"]="sampling.json"
    cmd = "cd ~/baseline-cvat300 && tar -czf - " + " ".join(files)
    data = subprocess.check_output(["ssh","-o","BatchMode=yes",args.ssh,cmd])
    with tarfile.open(fileobj=BytesIO(data),mode="r:gz") as archive:
        assert set(archive.getnames())==set(files) and len(archive.getnames())==len(files)
        bodies={local:archive.extractfile(name).read() for name,local in files.items()}
    compare=json.loads(bodies["comparison.json"])
    held=json.loads(bodies["held-score.json"])
    resources=json.loads(bodies["resources.json"])
    pack=bodies["submission.zip"]
    assert len(pack)==compare["zip_bytes"]<=42*2**20
    assert digest(pack)==compare["zip_sha256"]==held["zip_sha256"]
    assert digest(bodies["best.pt"])==compare["best_sha256"]
    with ZipFile(BytesIO(pack)) as z:
        assert z.namelist()==["model.onnx"] and digest(z.read("model.onnx"))==compare["onnx_sha256"]
    assert compare["validation_sha_verified"]==140 and compare["opsets"][""]==17
    parity=compare["onnx_parity"]
    assert len(parity)==6 and len({r["file"] for r in parity})==3
    assert {tuple(r["shape"]) for r in parity}=={(1,47,8400),(1,32,160,160)}
    assert held["holdout_sha256"]=="9e518defa9f4567814bd7f7d9a472130c3a0c7f05e0f54b6488e5bc6b2d4c0b0"
    metric=compare["scores"]["pilot"]
    assert metric["all140"]["map50_95"]==json.loads(bodies["run_report.json"])["val_estimate"]["map50_95"]
    board=json.loads(args.scoreboard.read_text())
    result={k:compare[k] for k in ("paired_scene_bootstrap_95_points","onnx_parity","zip_sha256","onnx_sha256","best_sha256","zip_bytes","opsets")}
    result.update(scores=metric, resources=resources, holdout=held)
    if args.run in board and result!=board[args.run]:
        raise RuntimeError("Scoreboard already contains a different record")
    folder=args.out/args.run
    folder.mkdir(exist_ok=True,parents=True)
    for name,body in bodies.items():
        target=folder/name
        if target.exists() and body!=target.read_bytes():
            raise RuntimeError(f"Local artifact differs: {target}")
        if not target.exists():
            temp=target.with_suffix(target.suffix+".partial")
            temp.write_bytes(body)
            temp.replace(target)
    board[args.run]=result
    temp=args.scoreboard.with_suffix(".partial")
    temp.write_text(json.dumps(board,indent=1)+"\n")
    temp.replace(args.scoreboard)
    print(args.run, {k:round(100*v["map50_95"],2) for k,v in metric.items()}, "holdout",held["score"], "ZIP",digest(pack))


if __name__=="__main__":
    main()
