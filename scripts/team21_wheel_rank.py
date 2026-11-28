"""Adapt the exported detector to occupied and parked two-wheelers on train scenes."""
import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms, box_convert, box_iou
from pycocotools import mask as coco

from notebook_identity import verified_team
from overnight_last_checkpoint import write
from team21_reached_target import main as reached_target
from team21_weight_mean import digest, read
from team21_wheel_masks import prepare

WHEEL = (4, 5, 6, 7)
PRIORITY = (4, 6, 7)


def sample_loss(pred, teacher, truth, flip):
    classes, boxes, _ = truth
    classes, boxes = classes.cuda(), boxes.cuda()
    if flip:
        boxes = boxes.clone()
        boxes[:, [0, 2]] = 640 - boxes[:, [2, 0]]
    rects = box_convert(pred[:, :4], "cxcywh", "xyxy")
    old_rects = box_convert(teacher[:, :4], "cxcywh", "xyxy")
    probs = pred[:, 4:15].clamp(.000001, .999999)
    prior = teacher[:, 4:15].clamp(.000001, .999999)
    with torch.no_grad():
        # The teacher fixes anchor assignments across epochs, including missed wheels.
        overlap = box_iou(boxes, old_rects)
        centers = teacher[:, :2]
        inside = ((centers[None] >= boxes[:, None, :2]) & (centers[None] <= boxes[:, None, 2:])).all(-1)
        affinity = overlap * inside * (.2 + prior[:, classes].T).pow(.18)
        quality, anchors = affinity.topk(5, dim=1)
        rows = torch.arange(len(boxes), device=boxes.device)[:, None].expand_as(anchors)
        rows, anchors = rows[quality > .18], anchors[quality > .18]
        selected = torch.isin(classes[rows], classes.new_tensor(WHEEL))
        rows, anchors = rows[selected], anchors[selected]
        # Preserve the classifier away from edited classes and around competing objects.
        keep = prior.amax(1).topk(160).indices
        hard = prior[:, list(WHEEL)].topk(40, dim=0).indices.flatten().unique()
    keep_cls = F.binary_cross_entropy(probs[keep], prior[keep])
    keep_boxes = F.smooth_l1_loss(rects[keep] / 16, old_rects[keep] / 16)
    if not len(anchors):
        return .4 * keep_cls + .18 * keep_boxes
    positive = probs[anchors][:, list(WHEEL)]
    categories = classes[rows] - 4
    target = torch.zeros_like(positive)
    target[torch.arange(len(rows), device=boxes.device), categories] = (
        .60 + .32 * overlap[rows, anchors]
    )
    weights = torch.where(classes[rows] == 5, .55, 1.)
    wheel_cls = (F.binary_cross_entropy(positive, target, reduction="none").sum(1) * weights).mean()
    normalizer = (boxes[rows, 2:] - boxes[rows, :2]).clamp_min(12)
    box = F.smooth_l1_loss((rects[anchors] - boxes[rows]) / normalizer.repeat(1, 2),
                           torch.zeros_like(rects[anchors]), beta=.12)
    with torch.no_grad():
        wheel_truth = torch.isin(classes, classes.new_tensor(WHEEL))
        background = hard[overlap[wheel_truth][:, hard].amax(0) < .20]
    negative = -torch.log1p(-probs[background][:, list(WHEEL)]).mean() if len(background) else 0.
    return wheel_cls + 1.5 * box + .75 * negative + .9 * keep_cls + .18 * keep_boxes


@torch.no_grad()
def check_metrics(net, data, ids):
    # Image-level NMS and one-to-one mask matches, measured only on the internal train fold.
    collected = {c: [] for c in WHEEL}
    counts = {c: 0 for c in WHEEL}
    axis = torch.arange(160, device="cuda") + .5
    thresholds = torch.tensor([.5, .65, .8], device="cuda")
    for index in ids:
        pixels, (cls, _, rings) = data[index]
        raw, proto = net(pixels[None].cuda().float()/255)
        raw = raw[0].T
        boxes = box_convert(raw[:, :4], "cxcywh", "xyxy").clamp(0, 640)
        scores, indices = raw[:, 4:15].flatten().topk(2200)
        anchors, category = indices // 11, indices % 11
        indices = batched_nms(boxes[anchors], scores, category, .7)[:100]
        for c in WHEEL:
            ids_c = indices[category[indices] == c]
            gt = [ring for label, ring in zip(cls.tolist(), rings) if label == c]
            counts[c] += len(gt)
            if not len(ids_c):
                continue
            chosen = anchors[ids_c]
            x1, y1, x2, y2 = (boxes[chosen].T / 4).unbind()
            inside = (axis >= x1[:, None]) & (axis < x2[:, None])
            mask = (raw[chosen, 15:] @ proto[0].flatten(1)).reshape(-1,160,160) > 0
            mask &= inside[:, None, :] & ((axis >= y1[:, None]) & (axis < y2[:, None]))[:, :, None]
            hits = torch.zeros((len(chosen), len(thresholds)), device="cuda")
            if gt:
                ground = torch.from_numpy(coco.decode(gt)).cuda().permute(2,0,1)[:,2::4,2::4].flatten(1).float()
                predicted = mask.flatten(1).float()
                overlap = predicted @ ground.T
                overlap /= (predicted.sum(1)[:,None] + ground.sum(1) - overlap).clamp_min(1)
                for col, threshold in enumerate(thresholds):
                    available = overlap.clone()
                    for i in range(len(chosen)):
                        quality, j = available[i].max(0)
                        if quality >= threshold:
                            hits[i,col] = 1
                            available[:,j] = -1
            collected[c].append(torch.cat([scores[ids_c,None],hits],1).cpu())
    aps = {}
    for c in WHEEL:
        if not counts[c] or not collected[c]:
            aps[c] = 0.
            continue
        items = torch.cat(collected[c])
        hit = items[items[:,0].argsort(descending=True),1:]
        precision = hit.cumsum(0)/torch.arange(1,len(hit)+1)[:,None]
        aps[c] = (precision*hit).sum(0).mean().item()/counts[c]
    return aps


def main():
    args = argparse.ArgumentParser()
    for name in ("parent","source","extra","out","notebook"):
        args.add_argument(name,type=Path)
    a = args.parse_args()
    if reached_target() == 0:
        return
    torch.manual_seed(21)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    report = read(a.parent/"run_report.json")
    assert verified_team(report,digest(a.notebook)) == 21
    pt = a.parent/"runs/train/weights/best.pt"
    assert digest(pt) == report["train"]["best_sha256"]
    checkpoint = torch.load(pt, map_location="cpu", weights_only=False)
    net = checkpoint.get("ema") or checkpoint["model"]
    net = net.cuda().float().eval().requires_grad_(False)
    head = net.model[-1]
    old = net.end2end, head.export, head.format
    net.end2end, head.export, head.format = False, True, "onnx"
    teacher = deepcopy(net)
    head.cv2.requires_grad_(True)
    head.cv3.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=.00010, weight_decay=.002)
    ns = {"__name__":"__main__"}
    for cell in read(a.notebook)["cells"][1:6]:
        exec("".join(cell["source"]),ns)
    with TemporaryDirectory() as temp:
        checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"],Path(temp))
        data, records = prepare(a.source,a.extra,a.parent,checker.infer.preprocess)
    train = [i for i,r in enumerate(records) if r["fold"]=="fit"]
    check = [i for i,r in enumerate(records) if r["fold"]=="check"]
    priority = [i for i in train if records[i]["focus"]]
    reviewed = [i for i in train if records[i]["reviewed"]]
    assert len(check) >= 40 and len(reviewed)==30
    def evaluate():
        metrics = check_metrics(net,data,check)
        return sum(metrics[c] for c in PRIORITY)/3, metrics
    baseline, initial = evaluate()
    print("Train-only baseline",initial, flush=True)
    history = []
    winner, best = None, baseline
    for epoch in range(10):
        if reached_target()==0:
            return
        order = [train[i] for i in torch.randperm(len(train))[:360]]
        order += [priority[i] for i in torch.randperm(len(priority))[:140]]+reviewed
        order = [order[i] for i in torch.randperm(len(order))]
        for j in range(0,len(order),5):
            batch = order[j:j+5]
            pixels = torch.stack([data[i][0] for i in batch]).cuda().float()/255
            flip = torch.rand(()).item() < .5
            if flip:
                pixels = pixels.flip(-1)
            with torch.no_grad():
                reference, _ = teacher(pixels)
            pred, _ = net(pixels)
            loss = torch.stack([sample_loss(row.T,parent.T,data[i][1],flip)
                                for row,parent,i in zip(pred,reference,batch,strict=True)]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,3.)
            opt.step()
        score, metrics = evaluate()
        history.append(metrics)
        print("Wheel rank",epoch+1,metrics, "mean",round(score,5),flush=True)
        if score > best:
            best = score
            winner = [p.detach().clone() for p in params]
            selected = epoch+1
    if winner is None:
        print("No train-only improvement",flush=True)
        return
    with torch.no_grad():
        for param, value in zip(params,winner,strict=True):
            param.copy_(value)
    net.end2end,head.export,head.format = old
    checkpoint.update(model=net.cpu().half(),ema=None,optimizer=None,epoch=-1)
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(checkpoint,a.out/"best.pt")
    write(a.out/"fit.json",dict(method="wheel-rank",parent=a.parent.name,parent_weights_sha256=digest(pt),
          weights_sha256=digest(a.out/"best.pt"),seed=21,source=a.source.name,extra_sha256=digest(a.extra/"manifest.json"),
          notebook_sha256=digest(a.notebook),images=records,baseline=initial,check_history=history,selected_epoch=selected))


if __name__ == "__main__":
    main()
