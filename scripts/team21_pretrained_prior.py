"""Interpolate a Team 21 fit with its COCO initialization in aligned parameter space."""
from pathlib import Path
import torch

from notebook_identity import verified_team
from team21_head_transfer import TARGETS
from team21_weight_mean import digest, read


# COCO annotates riders separately. Keep learned rows for all two-wheeler variants,
# Other vehicle and Sign, which has a wider scope than COCO stop sign.
COMMON = {0: "car", 1: "bus", 2: "truck", 8: "person", 10: "traffic light"}


@torch.no_grad()
def interpolate(parent, prior, fraction, out):
    if not 0 < fraction < 1:
        raise ValueError("Expected an interior interpolation fraction")
    resources = read(parent / "resources.json")
    report = read(parent / "run_report.json")
    assert verified_team(report, resources["notebook_sha256"]) == 21
    start = parent / "runs/train/weights/best.pt"
    original_sha = digest(prior)
    assert original_sha == "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
    assert digest(start) == report["train"]["best_sha256"]
    from ultralytics import YOLO
    src = YOLO(str(prior)).model.cpu().float()
    ckpt = torch.load(start, map_location="cpu", weights_only=False)
    dst = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    dst = dst.cpu().float()
    assert tuple(dst.names[i] for i in range(11)) == TARGETS
    assert len(src.names) == 80 and type(src.model[-1]) is type(dst.model[-1])
    indices = {name: index for index, name in src.names.items()}
    a, b = src.state_dict(), dst.state_dict()
    assert a.keys() == b.keys()
    remapped = []
    for key, value in b.items():
        source = a[key]
        if value.shape == source.shape:
            if value.is_floating_point():
                value.lerp_(source, fraction)
        else:
            assert key.startswith(("model.23.cv3.", "model.23.one2one_cv3.", "model.23.proto.semseg."))
            assert (source.shape[0], value.shape[0]) == (80, 11) and value.shape[1:] == source.shape[1:]
            for i, category in COMMON.items():
                value[i].lerp_(source[indices[category]], fraction)
            remapped.append(key)
        assert torch.isfinite(value).all()
    assert len(remapped) == 14
    ckpt.update(model=dst.half(), ema=None, optimizer=None, epoch=-1)
    out.parent.mkdir(exist_ok=True, parents=True)
    tmp = out.with_suffix(".partial")
    torch.save(ckpt, tmp)
    tmp.replace(out)
    return dict(prior_sha256=original_sha, prior_fraction=fraction, aligned_rows=list(COMMON.values()),
                team_run=parent.name, team_weights_sha256=digest(start), row_tensors=remapped)
