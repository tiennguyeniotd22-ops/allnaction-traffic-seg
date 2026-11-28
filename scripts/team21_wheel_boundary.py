"""Jointly fit bounding-box extent and the mask projection around observed train instances."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou, roi_align
from pycocotools import mask as coco

from team21_weight_mean import digest, read
from team21_wheel_masks import prepare
from team21_wheel_rank import check_metrics, PRIORITY
from team21_reached_target import main as reached_target
from notebook_identity import verified_team
from overnight_last_checkpoint import write


def objective(pred, proto, truth, flip):
    cls, boxes, rings = truth
    cls, boxes = cls.cuda(), boxes.cuda()
    if flip:
        boxes = boxes.clone()
        boxes[:, [0,2]] = 640-boxes[:, [2,0]]
    xyxy = box_convert(pred[:,:4], "cxcywh","xyxy").clamp(0,640)
    with torch.no_grad():
        overlap = box_iou(boxes,xyxy)
        scores = pred[:,4+cls].T
        affinity = overlap.square() * scores.clamp_min(.001).sqrt()
        value, anchors = affinity.topk(3,dim=1)
        rows = torch.arange(len(cls),device=cls.device)[:,None].expand_as(anchors)
        use = (value>.017) & (overlap.gather(1,anchors)>.36)
        rows, anchors = rows[use], anchors[use]
        if not len(rows):
            return None
        # Cover both complete silhouettes and predicted rectangles, including missed extremities.
        matched = boxes[rows]
        selected = xyxy[anchors].detach()
        l = torch.minimum(selected[:,:2],matched[:,:2])
        r = torch.maximum(selected[:,2:],matched[:,2:])
        roi = torch.cat([l,r],1)
        extent = (r-l).clamp_min(1.)
        decoded = coco.decode([rings[i] for i in rows.cpu().tolist()])
        bits = torch.from_numpy(decoded.transpose(2,0,1).copy()).cuda()[:,None].float()
        if flip:
            bits = bits.flip(-1)
        target = roi_align(bits,[b[None] for b in roi],(32,32),1,2,True)[:,0]
        weights = torch.where(torch.isin(cls[rows],cls.new_tensor(PRIORITY)),1.8,1.)
        weights *= (.5 + scores[rows,anchors].sqrt())
        grid = (torch.arange(32,device=cls.device)+.5)/32
        x = l[:,0,None]+grid*extent[:,0,None]
        y = l[:,1,None]+grid*extent[:,1,None]
    boxes_pred = xyxy[anchors]
    basis = roi_align(proto[None],[roi],(32,32),.25,2,True)
    logits = torch.einsum("nc,nchw->nhw",pred[anchors,15:],basis)
    x1,y1,x2,y2 = boxes_pred.T
    support_x = torch.sigmoid((x-x1[:,None])/1.6)*torch.sigmoid((x2[:,None]-x)/1.6)
    support_y = torch.sigmoid((y-y1[:,None])/1.6)*torch.sigmoid((y2[:,None]-y)/1.6)
    support = support_x[:,None,:]*support_y[:,:,None]
    prob = logits.sigmoid()*support
    union = (prob+target-prob*target).sum((1,2))
    iou = ((prob*target).sum((1,2))+1)/(union+1)
    # Supervise support outside the predicted box so small under-sized boxes can expand.
    bce = F.binary_cross_entropy_with_logits(logits,target,reduction="none").mean((1,2))
    delta = (boxes_pred-matched)/extent.repeat(1,2)
    regression = F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.06,reduction="none").mean(1)
    return ((1-iou+.35*bce+.8*regression)*weights).sum()/weights.sum()


def main():
    p=argparse.ArgumentParser()
    for k in ("parent","source","extra","out","notebook"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    if reached_target()==0:
        return
    torch.manual_seed(21)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    start=a.parent/"runs/train/weights/best.pt"
    report=read(a.parent/"run_report.json")
    assert verified_team(report,digest(a.notebook))==21 and digest(start)==report["train"]["best_sha256"]
    checkpoint=torch.load(start,map_location="cpu",weights_only=False)
    net=(checkpoint.get("ema") or checkpoint["model"]).float().cuda().eval().requires_grad_(False)
    head=net.model[-1]
    old=net.end2end,head.export,head.format
    net.end2end,head.export,head.format=False,True,"onnx"
    for tower in [*head.cv2,*head.cv4]:
        tower[-1].requires_grad_(True)
    head.proto.cv3.requires_grad_(True)
    params=[v for v in net.parameters() if v.requires_grad]
    ns={"__name__":"__main__"}
    for cell in read(a.notebook)["cells"][1:6]:
        exec("".join(cell["source"]),ns)
    with TemporaryDirectory() as temp:
        checker=ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"],Path(temp))
        data,records=prepare(a.source,a.extra,a.parent,checker.infer.preprocess)
    fit=[i for i,r in enumerate(records) if r["fold"]=="fit"]
    check=[i for i,r in enumerate(records) if r["fold"]=="check"]
    assert len(check)>=40 and len(fit)>=400
    opt=torch.optim.AdamW(params,lr=.00012,weight_decay=.001)
    def batch(indices,flip=False):
        pixels=torch.stack([data[i][0] for i in indices]).cuda().float()/255
        prediction,protos=net(pixels.flip(-1) if flip else pixels)
        losses=[objective(row.T,proto,data[i][1],flip) for i,row,proto in zip(indices,prediction,protos,strict=True)]
        losses=[loss for loss in losses if loss is not None]
        return torch.stack(losses).mean() if losses else None
    @torch.no_grad()
    def evaluate():
        aps=check_metrics(net,data,check)
        losses=[batch(check[j:j+4]) for j in range(0,len(check),4)]
        costs=[v.item() for v in losses if v is not None]
        metric=sum(aps[c] for c in PRIORITY)/3-sum(costs)/len(costs)*.45
        return metric,aps,sum(costs)/len(costs)
    best, initial, base_loss=evaluate()
    print("Boundary baseline",best,initial,base_loss,flush=True)
    history=[]
    winner=None
    for epoch in range(10):
        if reached_target()==0:
            return
        ids=[fit[i] for i in torch.randperm(len(fit))[:600]]
        ids.extend(i for i in fit if records[i]["reviewed"])
        ids=[ids[i] for i in torch.randperm(len(ids))]
        for j in range(0,len(ids),4):
            loss=batch(ids[j:j+4],torch.rand(()).item()<.5)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,3.)
            opt.step()
        metric,aps,cost=evaluate()
        history.append(dict(metric=metric,ap=aps,loss=cost))
        print("Boundary epoch",epoch+1,metric,aps,cost,flush=True)
        if metric>best:
            best=metric
            winner=[v.detach().clone() for v in params]
            selected=epoch+1
    if winner is None:
        print("No internal improvement",flush=True)
        return
    with torch.no_grad():
        for param,value in zip(params,winner,strict=True):
            param.copy_(value)
    net.end2end,head.export,head.format=old
    checkpoint.update(model=net.cpu().half(),ema=None,optimizer=None,epoch=-1)
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(checkpoint,a.out/"best.pt")
    write(a.out/"fit.json",dict(method="wheel-boundary",parent=a.parent.name,parent_weights_sha256=digest(start),
          weights_sha256=digest(a.out/"best.pt"),seed=21,source=a.source.name,extra_sha256=digest(a.extra/"manifest.json"),
          notebook_sha256=digest(a.notebook),images=records,initial_ap=initial,baseline_loss=base_loss,
          history=history,selected_epoch=selected))


if __name__=="__main__":
    main()
