"""Transfer only the changed tail of a ZIP with an identical prefix. Verify both full digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["pack", "restore"])
    p.add_argument("base", type=Path)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    folder = args.folder
    manifest = json.loads((folder / "manifest.json").read_bytes())
    assert digest(args.base) == manifest["base_sha256"]
    with ZipFile(args.base) as archive:
        prefix = archive.start_dir
    target = folder / "dataset.zip"
    tail = folder / "append.delta"
    if args.operation == "pack":
        assert digest(target) == manifest["dataset_sha256"]
        with args.base.open("rb") as base, target.open("rb") as full:
            for offset in range(0, prefix, 2**20):
                count = min(prefix-offset, 2**20)
                assert base.read(count) == full.read(count)
            with tail.open("wb") as delta:
                shutil.copyfileobj(full, delta)
    elif not target.exists():
        tmp = target.with_suffix(".partial")
        with args.base.open("rb") as base, tmp.open("wb") as out, tail.open("rb") as delta:
            for offset in range(0, prefix, 2**20):
                out.write(base.read(min(prefix-offset, 2**20)))
            shutil.copyfileobj(delta, out)
        assert digest(tmp) == manifest["dataset_sha256"]
        tmp.replace(target)
    assert digest(target) == manifest["dataset_sha256"]
    print(digest(target), tail.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
