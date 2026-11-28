"""Regenerate an overnight_extend ZIP from its unchanged base and reviewed additions."""
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
import hashlib
import json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    root = args.folder
    meta = json.loads((root / "manifest.json").read_bytes())
    assert digest(args.base) == meta["base_sha256"]
    dest = root / "dataset.zip"
    if not dest.exists():
        tmp = dest.with_suffix(".partial")
        with ZipFile(args.base) as base, ZipFile(root / "additions.zip") as changes, ZipFile(tmp, "w") as out:
            names = [n for row in meta["records"] for n in
                     ("images/train/" + row["stem"] + ".jpg", "labels/train/" + row["stem"] + ".txt")]
            assert changes.namelist() == names and not set(names).intersection(base.namelist())
            for archive in (base, changes):
                for name in archive.namelist():
                    info = ZipInfo(name, (2026, 11, 27, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    out.writestr(info, archive.read(name), compress_type=ZIP_DEFLATED)
        assert digest(tmp) == meta["dataset_sha256"]
        tmp.replace(dest)
    assert digest(dest) == meta["dataset_sha256"]
    print(meta["dataset_sha256"], meta["train"], meta["val"], flush=True)


if __name__ == "__main__":
    main()
