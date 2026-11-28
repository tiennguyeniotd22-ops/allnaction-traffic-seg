"""Fit a shared mask-coefficient transform on train ROIs, then fold it into the existing head."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou, roi_align

from team21_mask_quality import target_masks
from team21_weight_mean import digest, read


@torch.no_grad()
def sample(net, image, label, preprocess):
    tensor, meta = preprocess(image)
    truth = target_masks(label, meta)
    if truth is None:
        return None
    classes, gt_boxes, gt_masks = truth
    predictions, proto = net(torch.from_numpy(tensor.copy()))
    pred = predictions[0].T
    assert pred.shape == (8400, 47) and proto.shape == (1, 32, 160, 160)
    boxes = box_convert(pred[:, :4], "cxcywh", "xyxy").clamp(0, 640)
    iou = box_iou(gt_boxes, boxes)
    rank = pred[:, 4 + classes].T * iou.square()
    anchors = rank.argmax(1)
    rows = torch.arange(len(classes))
    keep = (iou[rows, anchors] >= .50) & (rank[rows, anchors] > .015)
    rows, anchors = rows[keep], anchors[keep]
    if not len(rows):
        return None
    rois = boxes[anchors] / 4
    options = dict(output_size=(20, 20), spatial_scale=1., sampling_ratio=2, aligned=True)
    p = roi_align(proto, [rois], **options).half()
    target = roi_align(gt_masks[rows, None].float(), [r[None] for r in rois], **options)
    return pred[anchors, 15:].contiguous(), p, target[:, 0].half()


def objective(transform, bias, batch):
    coeff, proto, target = batch
    logits = torch.einsum("nc,nchw->nhw", F.linear(coeff, transform, bias), proto.float())
    target = target.float()
    prob = logits.sigmoid()
    dice = (2 * (prob*target).sum((1, 2)) + 1) / ((prob + target).sum((1, 2)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1-dice).mean()


def fit(parent, output, preprocess):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    source = parent / "runs/train/weights/best.pt"
    report = read(parent / "run_report.json")
    assert report["team_id"] == report["seed"] == 21 and digest(source) == report["train"]["best_sha256"]
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    net = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    net = net.float().eval()
    head = net.model[-1]
    assert head.__class__.__name__ == "Segment26" and head.nc == 11
    orig_mode, orig_export, orig_format = net.end2end, head.export, head.format
    net.end2end, head.export, head.format = False, True, "onnx"
    data = parent / "dataset"
    excluded = {digest(p) for p in (data/"images/val").glob("*.jpg")}
    candidates = [(k,v) for k,v in read(data/"source_map.json")["files"].items()
                  if k.startswith("train/") and Path(v).parent.as_posix()=="images/train" and "_syn_" not in v]
    candidates.sort(key=lambda pair: hashlib.sha256(("21/mask-roi/"+pair[1]).encode()).digest())
    records, samples = [], [[], []]
    for i,(key,original) in enumerate(candidates[:850]):
        image, label = data/"images"/(key+".jpg"), data/"labels"/(key+".txt")
        checksum = digest(image)
        assert checksum not in excluded
        batch = sample(net, image, label, preprocess)
        fold = int(i % 8 == 0)
        if batch is not None:
            samples[fold].append(batch)
        records.append(dict(file=original, image_sha256=checksum, label_sha256=digest(label), fit=not fold))
        if len(records) % 100 == 0:
            print("Train frames processed",len(records),flush=True)
    batches = [tuple(torch.cat(parts) for parts in zip(*items, strict=True)) for items in samples]
    train, check = batches
    assert len(train[0]) >= 500 and len(check[0]) >= 50
    eye = torch.eye(32)
    transform = torch.nn.Parameter(eye.clone())
    bias = torch.nn.Parameter(torch.zeros(32))
    opt = torch.optim.AdamW([transform,bias],lr=.0007,weight_decay=0)
    @torch.no_grad()
    def evaluate():
        losses = []
        for ids in torch.arange(len(check[0])).split(96):
            losses.append(len(ids)*objective(transform,bias,[x[ids] for x in check]).item())
        return sum(losses)/len(check[0])
    baseline = best = evaluate()
    selected = None
    history = []
    for epoch in range(8):
        for ids in torch.randperm(len(train[0])).split(64):
            opt.zero_grad(set_to_none=True)
            loss = objective(transform,bias,[x[ids] for x in train])
            loss += .03*(transform-eye).square().sum()/32 + .03*bias.square().mean()
            loss.backward()
            opt.step()
        check_loss = evaluate()
        history.append(check_loss)
        if check_loss < best:
            best = check_loss
            selected = (epoch+1, transform.detach().clone(), bias.detach().clone())
    if selected is None:
        raise ValueError("Mask coefficients did not improve on the train-only internal check")
    epoch, matrix, offset = selected
    with torch.no_grad():
        for tower in head.cv4:
            projection = tower[-1]
            assert projection.kernel_size==(1,1) and projection.out_channels==32
            projection.weight.copy_((matrix @ projection.weight.flatten(1)).reshape_as(projection.weight))
            projection.bias.copy_(matrix @ projection.bias + offset)
    net.end2end, head.export, head.format = orig_mode, orig_export, orig_format
    ckpt.update(model=net.half(), ema=None, optimizer=None, epoch=-1)
    output.parent.mkdir(exist_ok=True, parents=True)
    tmp = output.with_suffix(".partial")
    torch.save(ckpt,tmp)
    tmp.replace(output)
    return dict(parent=parent.name, parent_weights_sha256=digest(source), seed=21, images=records,
                fit_instances=len(train[0]), internal_check_instances=len(check[0]),
                baseline_loss=baseline, selected_epoch=epoch, check_losses=history)
