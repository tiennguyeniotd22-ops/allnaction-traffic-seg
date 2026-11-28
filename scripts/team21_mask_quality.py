"""Fit the existing dense class projections to mask quality on Team 21 training images."""
import hashlib
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou, box_convert

from team21_weight_mean import digest, read


def target_masks(label, meta):
    classes, boxes, masks = [], [], []
    for line in label.read_text().splitlines():
        cls, *xy = line.split()
        points = np.asarray(xy, dtype=float).reshape(-1, 2)
        points *= [meta.new_w, meta.new_h]
        points += [meta.left, meta.top]
        ring = coco.frPyObjects([(points / 4).ravel().tolist()], 160, 160)[0]
        if coco.area(ring) < 4:
            continue
        classes.append(int(cls))
        boxes.append([*points.min(0), *points.max(0)])
        masks.append(coco.decode(ring))
    if not masks:
        return None
    return torch.tensor(classes), torch.tensor(boxes, dtype=torch.float32), torch.from_numpy(np.stack(masks))


@torch.no_grad()
def cache_examples(model, outputs, image, label, preprocess):
    tensor, meta = preprocess(image)
    truth = target_masks(label, meta)
    if truth is None:
        return []
    classes, gt_boxes, gt_masks = truth
    preds, proto = model(torch.from_numpy(tensor.copy()))
    assert preds.shape == (1, 47, 8400) and proto.shape == (1, 32, 160, 160)
    raw = preds[0].T
    boxes = box_convert(raw[:, :4], "cxcywh", "xyxy")
    result = []
    offset = 0
    axis = torch.arange(160) + .5
    for level, features in enumerate(outputs):
        flat = features[0].flatten(1).T
        batch = raw[offset:offset+len(flat)]
        rects = boxes[offset:offset+len(flat)]
        # Keep confident foreground/background examples and nearby anchors for missed classes.
        by_score = batch[:, 4:15].amax(1).topk(80).indices
        by_box = box_iou(gt_boxes, rects).topk(4, dim=1).indices.flatten()
        selected = torch.unique(torch.cat([by_score, by_box]))
        l, t, r, b = (rects[selected].T / 4).unbind()
        inside_x = (axis >= l[:, None]) & (axis < r[:, None])
        inside_y = (axis >= t[:, None]) & (axis < b[:, None])
        masks = (batch[selected, 15:] @ proto[0].flatten(1)).reshape(-1, 160, 160) > 0
        masks &= inside_x[:, None, :] & inside_y[:, :, None]
        pred_pixels = masks.flatten(1).float()
        gt_pixels = gt_masks.flatten(1).float()
        shared = pred_pixels @ gt_pixels.T
        iou = shared / (pred_pixels.sum(1)[:, None] + gt_pixels.sum(1) - shared).clamp_min(1)
        targets = torch.zeros(len(selected), 11)
        for cls in classes.unique():
            overlap = iou[:, classes == cls].amax(1)
            targets[:, cls] = ((overlap - .35) / .6).clamp(0, 1)
        result.append((flat[selected].clone(), targets, batch[selected, 4:15].clone()))
        offset += len(flat)
    assert offset == 8400
    return result


def fit(parent, output, preprocess):
    report = read(parent / "run_report.json")
    source = parent / "runs/train/weights/best.pt"
    assert digest(source) == report["train"]["best_sha256"] and report["team_id"] == report["seed"] == 21
    torch.manual_seed(21)
    torch.set_num_threads(1)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    net = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    net = net.float().eval()
    original_mode = net.end2end
    net.end2end = False
    head = net.model[-1]
    original_export, original_format = head.export, head.format
    head.export = True
    head.format = "onnx"
    assert head.nc == 11 and head.__class__.__name__ == "Segment26"
    outputs = [None] * 3
    hooks = []
    for i, tower in enumerate(head.cv3):
        def save_input(module, inputs, level=i):
            outputs[level] = inputs[0].detach()
        hooks.append(tower[-1].register_forward_pre_hook(save_input))
    data = parent / "dataset"
    mapping = read(data / "source_map.json")["files"]
    files = sorted((k, v) for k, v in mapping.items() if k.startswith("train/")
                   and Path(v).parent.as_posix() == "images/train" and "_syn_" not in v)
    files.sort(key=lambda pair: hashlib.sha256(("21/quality/" + pair[1]).encode()).digest())
    val_hashes = {digest(p) for p in (data / "images/val").glob("*.jpg")}
    records, cache = [], [[], [], []]
    for key, original in files[:720]:
        image = data / "images" / (key + ".jpg")
        label = data / "labels" / (key + ".txt")
        checksum = digest(image)
        assert checksum not in val_hashes
        for level, batch in enumerate(cache_examples(net, outputs, image, label, preprocess)):
            cache[level].append(batch)
        records.append(dict(image=original, image_sha256=checksum, label_sha256=digest(label)))
        if len(records) % 100 == 0:
            print("Cached train images", len(records), flush=True)
    for hook in hooks:
        hook.remove()
    assert len(records) >= 500
    stats = []
    for tower, examples in zip(head.cv3, cache, strict=True):
        features, target, previous = map(torch.cat, zip(*examples, strict=True))
        projection = tower[-1]
        assert torch.allclose(F.linear(features[:512], projection.weight.flatten(1), projection.bias).sigmoid(),
                              previous[:512], atol=2e-5, rtol=1e-4)
        projection.requires_grad_(True)
        # Keep half of the original probability to regularize the projection on unobserved scenes.
        desired = .5 * target + .5 * previous
        opt = torch.optim.AdamW(projection.parameters(), lr=.0008, weight_decay=.001)
        for epoch in range(8):
            for ids in torch.randperm(len(features)).split(1024):
                opt.zero_grad(set_to_none=True)
                scores = F.linear(features[ids], projection.weight.flatten(1), projection.bias)
                loss = F.binary_cross_entropy_with_logits(scores, desired[ids])
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores = F.linear(features, projection.weight.flatten(1), projection.bias)
            stats.append(dict(anchors=len(target), target_mass=target.sum(0).tolist(),
                              before=F.binary_cross_entropy(previous.clamp(.000001,.999999), desired).item(),
                              after=F.binary_cross_entropy_with_logits(scores, desired).item()))
            assert np.isfinite(stats[-1]["after"]) and stats[-1]["after"] < stats[-1]["before"]
    head.export, head.format = original_export, original_format
    net.end2end = original_mode
    ckpt.update(model=net.half(), ema=None, optimizer=None, epoch=-1)
    output.parent.mkdir(exist_ok=True, parents=True)
    tmp = output.with_suffix(".partial")
    torch.save(ckpt, tmp)
    tmp.replace(output)
    return dict(parent=parent.name, parent_weights_sha256=digest(source), seed=21, epochs=8,
                target_original_fraction=.5, images=records, level_stats=stats)
