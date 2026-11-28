"""Supervise the existing dense segmentation heads on native contours in two-wheeler scenes."""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pycocotools import mask as coco
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou, roi_align

from notebook_identity import verified_team
from team21_reached_target import main as reached_target
from team21_weight_mean import digest, read
from overnight_last_checkpoint import write


def targets(body, meta):
    rings, classes, boxes = [], [], []
    for line in body.decode().splitlines():
        category,*tokens=line.split()
        xy=np.asarray(tokens,float).reshape(-1,2)*[meta.new_w,meta.new_h]+[meta.left,meta.top]
        rle=coco.frPyObjects([xy.ravel().tolist()],640,640)[0]
        if coco.area(rle)<64:
            continue
        rings.append(rle)
        classes.append(int(category))
        boxes.append([*xy.min(0),*xy.max(0)])
    if rings:
        return torch.tensor(classes),torch.tensor(boxes,dtype=torch.float32),rings


def prepare(source, extra, parent, preprocess):
    d=source/"dataset"
    mapping=read(d/"source_map.json")["files"]
    files=[(v,d/"images"/(k+".jpg"),d/"labels"/(k+".txt"),False) for k,v in mapping.items()
           if k.startswith("train/") and Path(v).parent.as_posix()=="images/train" and "_syn_" not in v]
    reference={digest(p) for p in (parent/"dataset/images/val").glob("*.jpg")}
    assert len(reference)==140 and reference=={digest(p) for p in (d/"images/val").glob("*.jpg")}
    for item in read(extra/"manifest.json"):
        k=item["key"]
        image,label=extra/"images"/(k+".jpg"),extra/"labels"/(k+".txt")
        assert digest(image)==item["image_sha256"] and digest(label)==item["label_sha256"]
        files.append((k,image,label,True))
    data,records=[],[]
    seen=set()
    for name,image,label,reviewed in files:
        body=label.read_bytes()
        cls={int(t.split()[0]) for t in body.splitlines()}
        focus=bool({4,6,7}&cls)
        rank=hashlib.sha256(("21/wheel-contour/"+name).encode()).digest()
        if not focus and not reviewed and rank[0]>85:
            continue
        checksum=digest(image)
        assert checksum not in reference and checksum not in seen
        seen.add(checksum)
        tensor,meta=preprocess(image)
        truth=targets(body,meta)
        if truth is None:
            continue
        # Keep the edited COCO cases in train. Split the internal check by source image.
        fold="check" if not reviewed and rank[1]<27 else "fit"
        data.append((torch.from_numpy(np.rint(tensor[0]*255).astype("uint8")),truth))
        records.append(dict(file=name,image_sha256=checksum,label_sha256=digest(label),reviewed=reviewed,focus=focus,fold=fold))
    return data,records


def loss_image(raw, proto, truth):
    classes,gt_boxes,rings=truth
    device=raw.device
    classes,gt_boxes=classes.to(device),gt_boxes.to(device)
    with torch.no_grad():
        boxes=box_convert(raw[:,:4].detach(),"cxcywh","xyxy").clamp(0,640)
        overlaps=box_iou(gt_boxes,boxes)
        cls_score=raw[:,4+classes].detach().T
        # Use high-overlap anchors even when an empty vehicle was classified as occupied.
        wheel_score=raw[:,8:12].detach().amax(1)
        wheels=(classes>=4)&(classes<=7)
        family_score=torch.where(wheels[:,None],torch.maximum(wheel_score,cls_score),cls_score)
        rank=overlaps.square()*(family_score+.03).sqrt()
        anchors=rank.argmax(1)
        rows=torch.arange(len(classes),device=device)
        use=(overlaps[rows,anchors]>=.45)&(rank[rows,anchors]>.028)
        rows,anchors=rows[use],anchors[use]
        if not len(rows):
            return None
        rects=boxes[anchors]
        subset=coco.decode([rings[i] for i in rows.cpu().tolist()])
        masks=torch.from_numpy(subset.transpose(2,0,1).copy()).to(device)[:,None].float()
        gt=roi_align(masks,[box[None] for box in rects],(40,40),1.,2,True)[:,0]
        weight=torch.where(wheels[rows],2.,1.)
    basis=roi_align(proto[None],[rects],(40,40),.25,2,True)
    logits=torch.einsum("nc,nchw->nhw",raw[anchors,15:],basis)
    prob=logits.sigmoid()
    dice=(2*(prob*gt).sum((1,2))+1)/((prob+gt).sum((1,2))+1)
    losses=F.binary_cross_entropy_with_logits(logits,gt,reduction="none").mean((1,2))+1-dice
    return (losses*weight).sum()/weight.sum()


def main():
    p=argparse.ArgumentParser()
    for name in ("parent","source","extra","out","notebook"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if reached_target()==0:
        return
    torch.manual_seed(21)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    report=read(a.parent/"run_report.json")
    assert verified_team(report,digest(a.notebook))==21
    start=a.parent/"runs/train/weights/best.pt"
    assert digest(start)==report["train"]["best_sha256"]
    checkpoint=torch.load(start,map_location="cpu",weights_only=False)
    net=checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    net=net.float().eval().cuda().requires_grad_(False)
    head=net.model[-1]
    mode,export,fmt=net.end2end,head.export,head.format
    net.end2end,head.export,head.format=False,True,"onnx"
    cells=read(a.notebook)["cells"]
    ns={"__name__":"__main__"}
    for i in (1,2,3,4,5):
        exec("".join(cells[i]["source"]),ns)
    with TemporaryDirectory() as tmp:
        checker=ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"],Path(tmp))
        data,records=prepare(a.source,a.extra,a.parent,checker.infer.preprocess)
    fit=[i for i,r in enumerate(records) if r["fold"]=="fit"]
    check=[i for i,r in enumerate(records) if r["fold"]=="check"]
    assert len(check)>=40 and len(fit)>=400
    groups=[head.cv4,head.proto.cv3,head.proto.cv2]
    for module in groups:
        module.requires_grad_(True)
    named={k:v for k,v in net.named_parameters() if v.requires_grad}
    before={k:v.detach().clone() for k,v in named.items()}
    opt=torch.optim.AdamW(named.values(),lr=.00028,weight_decay=.002)
    def batch(ids):
        images=torch.stack([data[i][0] for i in ids]).cuda().float()/255
        pred,proto=net(images)
        losses=[loss_image(raw.T,basis,data[i][1]) for i,raw,basis in zip(ids,pred,proto,strict=True)]
        usable=[x for x in losses if x is not None]
        return torch.stack(usable).mean() if usable else None
    @torch.no_grad()
    def evaluate():
        values=[]
        for j in range(0,len(check),4):
            loss=batch(check[j:j+4])
            if loss is not None:
                values.append(loss.item())
        return sum(values)/len(values)
    best=baseline=evaluate()
    winner=None
    history=[]
    for epoch in range(8):
        if reached_target()==0:
            return
        order=[fit[i] for i in torch.randperm(len(fit)).tolist()[:600]]
        # One extra pass over the newly reviewed examples in each epoch.
        order += [i for i in fit if records[i]["reviewed"]]
        order=[order[i] for i in torch.randperm(len(order)).tolist()]
        for j in range(0,len(order),4):
            opt.zero_grad(set_to_none=True)
            cost=batch(order[j:j+4])
            if cost is None:
                continue
            cost.backward()
            torch.nn.utils.clip_grad_norm_(named.values(),5.)
            opt.step()
        metric=evaluate()
        history.append(metric)
        if metric<best:
            winner={k:v.detach().clone() for k,v in named.items()}
            best=metric
        print("Wheel masks epoch",epoch+1,"train-only check",round(metric,5),"baseline",round(baseline,5),flush=True)
    if winner is None:
        raise RuntimeError("No internal-check improvement for wheel masks")
    with torch.no_grad():
        for name,value in named.items():
            value.copy_(winner[name])
    net.end2end,head.export,head.format=mode,export,fmt
    checkpoint.update(model=net.cpu().half(),ema=None,optimizer=None,epoch=-1)
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(checkpoint,a.out/"best.pt")
    write(a.out/"fit.json",dict(method="wheel-contours",parent=a.parent.name,parent_weights_sha256=digest(start),
          weights_sha256=digest(a.out/"best.pt"),seed=21,source=a.source.name,extra_sha256=digest(a.extra/"manifest.json"),
          notebook_sha256=digest(a.notebook),images=records,baseline_loss=baseline,check_losses=history,
          selected_epoch=history.index(best)+1))


if __name__=="__main__":
    main()
