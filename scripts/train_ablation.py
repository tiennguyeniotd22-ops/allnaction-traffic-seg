"""Training-only experiments for the frozen notebook and its paired evaluator."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import yaml


# Each recipe changes one intervention relative to the same baseline.
CHANGES = {
    "control": {},
    "repeat-rare": {},
    "no-copy-paste": {"copy_paste": (0.35, 0.0)},
    "mask-ratio-4": {"mask_ratio": (2, 4)},
    "lower-lr": {"lr0": (0.002, 0.0007)},
    "head-transfer": {"lr0": (0.002, 0.0007)},
    "sgd": {"optimizer": ("AdamW", "SGD")},
    "linear-lr": {"cos_lr": (True, False)},
}


def repeat_rare(data_yaml, seed, run):
    """Bound static repeat factors by two and retain every original train image."""
    path = Path(data_yaml)
    data = yaml.safe_load(path.read_text())
    root = Path(data["path"]).resolve()
    assert root == path.parent.resolve() and data["train"] == "images/train"
    files = sorted((root / data["train"]).glob("*.jpg"))
    counts = Counter()
    labels = {}
    for image in files:
        lines = (root / "labels/train" / (image.stem + ".txt")).read_text().splitlines()
        classes = {int(line.split()[0]) for line in lines}
        counts.update(classes)
        labels[image] = classes
    # Repeat the image at most once. Fix fractional rounding with the team's seed.
    rng = random.Random(seed)
    entries = []
    extra = []
    for image in files:
        factor = max((math.sqrt(0.12 * len(files) / counts[c]) for c in labels[image]), default=1)
        entries.append("./" + image.relative_to(root).as_posix())
        if rng.random() < min(1, max(0, factor - 1)):
            extra.append(entries[-1])
    if not extra:
        raise ValueError("No train images selected for repetition")
    listing = root / "train-repeat.txt"
    listing.write_text("\n".join(entries + extra) + "\n", encoding="utf8")
    data["train"] = listing.name
    config = root / "repeat.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    # Check the installed loader resolves the paths and retains their multiplicity.
    from ultralytics.data.base import BaseDataset
    from types import SimpleNamespace
    checked = BaseDataset.get_img_files(SimpleNamespace(fraction=1.0, prefix="repeat "), str(listing))
    assert Counter(checked) == Counter(str(root / entry) for entry in entries + extra)
    (run / "sampling.json").write_text(json.dumps(dict(
        seed=seed, threshold=0.12, maximum_repeat=2, unique_train=len(files),
        draws_per_epoch=len(checked), image_frequency=counts, repeats=extra,
        list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest()), indent=2))
    return str(config)


def configure_train_args(train_args, experiment, run):
    for name, (old, new) in CHANGES[experiment].items():
        if train_args[name] != old:
            raise ValueError(f"{name} differs from reference {old!r}")
        train_args[name] = new
    if experiment == "repeat-rare":
        train_args["data"] = repeat_rare(train_args["data"], train_args["seed"], run)
