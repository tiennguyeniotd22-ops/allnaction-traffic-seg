"""Adapt batch-normalization buffers to full-frame traffic images from the training split."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@torch.no_grad()
def refresh(ckpt_path, run, momentum=.025, platform_input=None):
    if not 0 < momentum <= 1:
        raise ValueError("BatchNorm momentum must be between zero (exclusive) and one")
    source = json.loads((run / "dataset/source_map.json").read_bytes())["files"]
    groups = {"Vietnam": [], "BDD": [], "COCO": []}
    for key, value in source.items():
        if not key.startswith("train/"):
            continue
        name = Path(value).stem
        group = ("Vietnam" if name.startswith("vn_sign_") else "BDD" if name.startswith("bdd_")
                 else "COCO" if name.startswith("zz_drive_") and "_syn_" not in name else None)
        if group:
            rank = hashlib.sha256(("21/" + value).encode()).hexdigest()
            groups[group].append((rank, key, value))
    count = min(64, *(len(items) for items in groups.values()))
    count -= count % 8
    assert count >= 32
    chosen = sorted(item for items in groups.values() for item in sorted(items)[:count])
    excluded = {hashlib.sha256(p.read_bytes()).digest() for p in (run / "dataset/images/val").glob("*.jpg")}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = ckpt["model"].float().eval()
    norm = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d) and m.track_running_stats]
    assert len(norm) > 20
    before = {k: tensor.clone() for k, tensor in net.state_dict().items()}
    for layer in norm:
        layer.train()
        layer.momentum = momentum
    records = []
    batch = []
    for _, key, value in chosen:
        file = run / "dataset/images/train" / (Path(key).name + ".jpg")
        body = file.read_bytes()
        digest = hashlib.sha256(body)
        assert digest.digest() not in excluded and Path(value).parent.as_posix() == "images/train"
        if platform_input is None:
            with Image.open(file) as original:
                image = original.convert("RGB")
            scale = min(640 / image.width, 640 / image.height)
            resized = image.resize(tuple(max(1, round(n * scale)) for n in image.size), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (640, 640), (114, 114, 114))
            canvas.paste(resized, ((640 - resized.width) // 2, (640 - resized.height) // 2))
            batch.append(np.asarray(canvas))
        else:
            tensor, _ = platform_input(file)
            assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
            batch.append(tensor)
        records.append(dict(file=value, sha256=digest.hexdigest()))
        if len(batch) == 8:
            if platform_input is None:
                inputs = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255)
            else:
                inputs = torch.from_numpy(np.concatenate(batch)).contiguous(memory_format=torch.channels_last)
            net(inputs)
            batch.clear()
    assert not batch
    net.eval()
    changed = []
    for name, tensor in net.state_dict().items():
        if not torch.equal(before[name], tensor):
            assert name.endswith(("running_mean", "running_var", "num_batches_tracked")) and torch.isfinite(tensor).all()
            changed.append(name)
    assert changed
    ckpt["model"] = net.half()
    tmp = ckpt_path.with_suffix(".bn-partial")
    with tmp.open("wb") as stream:
        torch.save(ckpt, stream)
    tmp.replace(ckpt_path)
    record = dict(seed=21, images=records, images_per_domain=count, domains=list(groups),
                  batch=8, momentum=momentum, updated_buffers=len(changed))
    if platform_input is not None:
        record["preprocessing"] = "notebook"
    return record
