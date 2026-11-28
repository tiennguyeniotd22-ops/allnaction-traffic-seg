"""Fit the dense classifier for four two-wheeler classes using training annotations only."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou

from team21_mask_quality import target_masks
from team21_weight_mean import digest, read


WHEELS = (4, 5, 6, 7)


@torch.no_grad()
def examples(net, hooks, tensor, truth):
    classes, gt_boxes, masks = truth
    pred, proto = net(torch.from_numpy(tensor.copy()))
    raw = pred[0].T
    boxes = box_convert(raw[:,:4],"cxcywh","xyxy").clamp(0,640)
    overlaps = box_iou(gt_boxes,boxes)
    result = []
    offset = 0
    for feature in hooks:
        flat = feature[0].flatten(1).T
        n = len(flat)
        prob = raw[offset:offset+n,4:15]
        overlap = overlaps[:,offset:offset+n]
        candidates = [prob[:,list(WHEELS)].topk(16,dim=0).indices.flatten()]
        positive = (classes[:,None] == torch.tensor(WHEELS)).any(1)
        if positive.any():
            q = overlap[positive] * (prob[:,list(WHEELS)].amax(1)+.1).sqrt()
            candidates.append(q.topk(6,dim=1).indices.flatten())
        ids = torch.unique(torch.cat(candidates))
        # Compare only with wheel GT, with mask IoU calculated inside each predicted box.
        subset = boxes[offset+ids]
        points = torch.arange(160)+.5
        left,top,right,bottom = (subset.T/4).unbind()
        inside = ((points>=left[:,None]) & (points<right[:,None]))[:,None,:]
        inside = inside & ((points>=top[:,None]) & (points<bottom[:,None]))[:,:,None]
        mask = (raw[offset+ids,15:] @ proto[0].flatten(1)).reshape(-1,160,160) > 0
        predmask = (mask&inside).flatten(1).float()
        target = torch.zeros(len(ids),4)
        if positive.any():
            ground = masks[positive].flatten(1).float()
            shared = predmask @ ground.T
            miou = shared/(predmask.sum(1)[:,None]+ground.sum(1)-shared).clamp_min(1)
            for j,c in enumerate(WHEELS):
                group = classes[positive]==c
                if group.any():
                    target[:,j] = miou[:,group].amax(1)
        result.append((flat[ids].clone(),prob[ids][:,list(WHEELS)].clone(),target))
        offset += n
    assert offset == 8400
    return result


def ranking(scores, quality):
    # Anchor-level AP on a train-only check. The official final score is obtained separately.
    values = []
    for c in range(4):
        sort = scores[:,c].argsort(descending=True)
        for threshold in (.5,.65,.8):
            truth = (quality[sort,c]>=threshold).float()
            if truth.sum() >= 5:
                precision = truth.cumsum(0)/torch.arange(1,len(truth)+1)
                values.append((precision*truth).sum()/truth.sum())
    return torch.stack(values).mean().item() if values else 0.


def fit(parent, data_parent, output, preprocess):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    source = parent/"runs/train/weights/best.pt"
    report = read(parent/"run_report.json")
    data_report = read(data_parent/"run_report.json")
    assert report["team_id"] == report["seed"] == data_report["team_id"] == data_report["seed"] == 21
    assert report["classes"] == data_report["classes"] and digest(source) == report["train"]["best_sha256"]
    train = data_parent/"dataset"
    excluded = {digest(p) for p in (parent/"dataset/images/val").glob("*.jpg")}
    assert excluded == {digest(p) for p in (train/"images/val").glob("*.jpg")} and len(excluded)==140
    ckpt = torch.load(source,map_location="cpu",weights_only=False)
    net = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    net = net.float().eval()
    head = net.model[-1]
    mode,export,fmt = net.end2end,head.export,head.format
    net.end2end,head.export,head.format=False,True,"onnx"
    features = [None]*3
    handles = []
    for i,tower in enumerate(head.cv3):
        def save(module,args,level=i):
            features[level]=args[0].detach()
        handles.append(tower[-1].register_forward_pre_hook(save))
    records = []
    folds = [[[],[],[]],[[],[],[]]]
    for key,original in sorted(read(train/"source_map.json")["files"].items()):
        if not key.startswith("train/") or "_syn_" in original:
            continue
        label=train/"labels"/(key+".txt")
        foreground=any(int(line.split()[0]) in WHEELS for line in label.read_text().splitlines())
        rank=hashlib.sha256(("21/wheel-head/"+original).encode()).digest()
        if not foreground and rank[0] >= 48:
            continue
        image=train/"images"/(key+".jpg")
        checksum=digest(image)
        assert checksum not in excluded and Path(original).parent.as_posix()=="images/train"
        tensor,meta=preprocess(image)
        truth=target_masks(label,meta)
        if truth is None:
            continue
        fold=int(rank[1]<32)
        for level,batch in enumerate(examples(net,features,tensor,truth)):
            folds[fold][level].append(batch)
        records.append(dict(file=original,image_sha256=checksum,label_sha256=digest(label),fit=not fold))
        if len(records)%200==0:
            print("Wheel train frames",len(records),flush=True)
    for hook in handles:
        hook.remove()
    stats=[]
    for level,tower in enumerate(head.cv3):
        features,original,quality=[torch.cat(x) for x in zip(*folds[0][level],strict=True)]
        check,prior,targets=[torch.cat(x) for x in zip(*folds[1][level],strict=True)]
        proj=tower[-1]
        start=proj.weight.detach().flatten(1)[list(WHEELS)].clone()
        initial_bias=proj.bias.detach()[list(WHEELS)].clone()
        weights=torch.nn.Parameter(start.clone())
        bias=torch.nn.Parameter(initial_bias.clone())
        assert torch.allclose(F.linear(check,weights,bias).sigmoid(),prior,atol=1e-5)
        opt=torch.optim.AdamW([weights,bias],lr=.0006,weight_decay=0)
        best=baseline=ranking(prior,targets)
        winner=(start,initial_bias,0)
        overlap=((quality-.35)/.55).clamp(0,1)
        desired=.65*overlap+.35*original
        # Emphasize true instances and the competing two-wheeler classes at the same location.
        importance=1+4*quality.amax(1,keepdim=True)
        for epoch in range(12):
            for ids in torch.randperm(len(features)).split(512):
                logits=F.linear(features[ids],weights,bias)
                loss=(importance[ids]*F.binary_cross_entropy_with_logits(logits,desired[ids],reduction="none")).mean()
                loss+=.004*(weights-start).square().sum()/4
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                metric=ranking(F.linear(check,weights,bias),targets)
                if metric>best:
                    best=metric
                    winner=(weights.detach().clone(),bias.detach().clone(),epoch+1)
        w,b,epoch=winner
        with torch.no_grad():
            proj.weight[list(WHEELS)]=w[:,:,None,None]
            proj.bias[list(WHEELS)]=b
        stats.append(dict(level=level,train_anchors=len(features),check_anchors=len(check),baseline=baseline,best=best,epoch=epoch))
    net.end2end,head.export,head.format=mode,export,fmt
    ckpt.update(model=net.half(),ema=None,optimizer=None,epoch=-1)
    output.parent.mkdir(exist_ok=True,parents=True)
    tmp=output.with_suffix(".partial")
    torch.save(ckpt,tmp)
    tmp.replace(output)
    return dict(parent=parent.name,parent_weights_sha256=digest(source),data_run=data_parent.name,
                dataset_sha256=read(data_parent/"resources.json")["dataset_sha256"],classes=WHEELS,
                method="wheel-classifier",seed=21,images=records,level_stats=stats)
