"""Evaluate the team's learned one-to-one head in the standard dense export."""
import torch

from team21_weight_mean import digest, read


@torch.no_grad()
def transfer(parent, output):
    source = parent / "runs/train/weights/best.pt"
    source_hash = digest(source)
    report = read(parent / "run_report.json")
    assert report["train"]["best_sha256"] == source_hash and report["team_id"] == report["seed"] == 21
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    head = model.model[-1]
    assert head.__class__.__name__ == "Segment26" and head.nc == 11
    assert list(model.names.values()) == report["classes"]
    branches = ["cv2", "cv3", "cv4"]
    tensors = 0
    for name in branches:
        src = getattr(head, "one2one_" + name).state_dict()
        dst = getattr(head, name)
        old = dst.state_dict()
        assert src.keys() == old.keys()
        assert any(not torch.equal(value, old[key]) for key, value in src.items())
        dst.load_state_dict(src, strict=True)
        tensors += len(src)
    checkpoint.update(model=model, ema=None, optimizer=None, epoch=-1)
    output.parent.mkdir(exist_ok=True, parents=True)
    tmp = output.with_suffix(".partial")
    torch.save(checkpoint, tmp)
    tmp.replace(output)
    return dict(parent=parent.name, source_weights_sha256=source_hash,
                source_prefix="one2one_", target_branches=branches, copied_tensors=tensors)
