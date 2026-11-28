"""Average aligned Team 21 checkpoints with equal weights, without using validation images."""
import hashlib
import json
from pathlib import Path

import torch

from notebook_identity import verified_team


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def mean_checkpoint(folders, output):
    if len(set(folders)) != len(folders) or len(folders) < 2:
        raise ValueError("Expected at least two distinct runs")
    records, states = [], []
    shared = None
    for folder in folders:
        file = folder / "runs/train/weights/best.pt"
        report = read(folder / "run_report.json")
        res = read(folder / "resources.json")
        assert verified_team(report, res["notebook_sha256"]) == 21
        assert digest(file) == report["train"]["best_sha256"]
        assert report["model"]["architecture"] == "yolo26n-seg"
        assert res["experiment"] in {"lower-lr", "sgd"}, "Only combine runs sharing the standard COCO initialization"
        identity = (res["initial_weights_sha256"], res["notebook_sha256"], report["environment"]["packages"]["ultralytics"],
                    report["classes"], report["model"]["nc"])
        if shared is None:
            shared = identity
        assert shared == identity and identity[-1] == 11
        record = dict(run=folder.name, dataset_sha256=res["dataset_sha256"],
                      weights_sha256=report["train"]["best_sha256"], revision=res["revision"],
                      experiment=res["experiment"], epochs=report["train"]["train_args"]["epochs"],
                      patience=report["train"]["train_args"]["patience"])
        records.append(record)
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        module = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
        assert list(module.names.values()) == report["classes"]
        states.append(module.float().state_dict())
        if len(states) == 1:
            base_model, base_checkpoint = module, ckpt
    assert all(state.keys() == states[0].keys() for state in states)
    for name, value in states[0].items():
        items = [state[name] for state in states]
        assert all(tensor.shape == value.shape and tensor.dtype == value.dtype for tensor in items)
        if value.is_floating_point():
            average = torch.stack(items).mean(dim=0)
            assert torch.isfinite(average).all()
            value.copy_(average)
    base_checkpoint.update(model=base_model.half(), ema=None, optimizer=None, epoch=-1)
    output.parent.mkdir(exist_ok=True, parents=True)
    tmp = output.with_suffix(".partial")
    with tmp.open("wb") as stream:
        torch.save(base_checkpoint, stream)
    tmp.replace(output)
    return records
