"""Measure a frozen, secondary diagnostic split with the competition notebook."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for k in ("pack", "manifest", "notebook", "submission", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    lock = json.loads(a.manifest.read_bytes())
    checksum = sha(a.submission.read_bytes())
    assert sha(a.pack.read_bytes()) == lock["zip_sha256"]
    nb = a.notebook.read_bytes()
    assert sha(nb) == "83e498336ed0a6081753ae49dc1dc2b93ca4066e28515a3204c799a9904876fd"
    result = a.out / "score.json"
    if result.exists():
        record = json.loads(result.read_bytes())
        if (record["zip_sha256"], record["probe_sha256"], record["manifest_sha256"]) == (checksum, lock["zip_sha256"], sha(a.manifest.read_bytes())):
            print(record["score"], flush=True)
            return
        raise ValueError("Existing score belongs to a different artifact")
    a.out.mkdir(parents=True, exist_ok=True)
    with ZipFile(a.pack) as z:
        assert sorted(z.namelist()) == sorted(lock["files"])
        for name, expected in lock["files"].items():
            data = z.read(name)
            assert sha(data) == expected
            path = a.out / name
            assert path.parent in (a.out / "images", a.out / "labels")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
    images = sorted((a.out / "images").glob("*.jpg"))
    assert {f.stem for f in images} == {r["key"] for r in lock["records"]}
    assert {f.stem for f in (a.out / "labels").glob("*.txt")} == {f.stem for f in images}
    ns = {"__name__": "__main__"}
    cells = json.loads(nb)["cells"]
    for i in (1, 2, 3, 4):
        exec(compile("".join(cells[i]["source"]), f"notebook-{i}", "exec"), ns)
    checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"], a.out / "_checker")
    predictions, _ = ns["run_platform_inference"](a.submission, images, a.out, checker)
    score = ns["score_predictions"](predictions, a.out / "labels", images, checker).to_dict()
    score.update(zip_sha256=checksum, probe_sha256=lock["zip_sha256"], manifest_sha256=sha(a.manifest.read_bytes()), notebook_sha256=sha(nb))
    assert score["n_images"] == len(images)
    temp = result.with_suffix(".partial")
    temp.write_text(json.dumps(score, indent=1) + "\n")
    temp.replace(result)
    print(score["score"], score["n_images"], flush=True)


if __name__ == "__main__":
    main()
