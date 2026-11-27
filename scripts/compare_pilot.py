"""Compare round 3 and the geometry pilot on the frozen competition validation set."""
import argparse
from collections import defaultdict
from importlib import metadata
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from ultralytics import YOLO
import yaml

from mask_geometry import polygon_area_at_640
from train_ablation import CHANGES


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for k in ("baseline","pilot","manifest","out","notebook"):
        p.add_argument("--"+k, required=True, type=Path)
    p.add_argument("--allow-schedule-change", action="store_true")
    p.add_argument("--experiment", choices=CHANGES, default="control")
    args = p.parse_args()
    ns = {"__name__":"__main__"}
    for i in (1,2,3,4):
        exec(compile("".join(read(args.notebook)["cells"][i]["source"]),f"notebook-{i}","exec"),ns)
    checker = ns["load_selfcheck"](ns["SELFCHECK_BUNDLE"],args.out.parent/"compare_checker")
    sc = checker.score
    params = sc.ScoringParams.from_config(checker.config)
    manifest = {r["file"]:r for r in read(args.manifest)}
    scores, configs, evaluated = {}, [], {}
    for key,run in [("round3",args.baseline),("pilot",args.pilot)]:
        images = sorted((run/"dataset/images/val").glob("*.jpg"))
        labels = run/"dataset/labels/val"
        files = read(run/"dataset/source_map.json")["files"]
        assert len(images)==len(manifest)==140
        assert {Path(files["val/"+i.stem]).name for i in images} == manifest.keys()
        groups = defaultdict(list)
        scenes = defaultdict(lambda:defaultdict(list))
        for index,img in enumerate(images):
            item = manifest[Path(files["val/"+img.stem]).name]
            assert sha(img.read_bytes()) == item["image_sha256"]
            # The notebook normalizes label serialization; compare the raw input to the
            # manifest, and check the normalized GT used by both scorers byte-for-byte.
            raw = run/"raw/labels/val"/(Path(item["file"]).stem+".txt")
            assert sha(raw.read_bytes()) == item["labels_sha256"]
            assert (labels/(img.stem+".txt")).read_bytes() == (args.baseline/"dataset/labels/val"/(img.stem+".txt")).read_bytes()
            groups[item["source"]].append(index)
            groups[item["dataset"]].append(index)
            scenes[item["condition"]][item["scene"]].append(index)
        pred = sc._as_predictions(read(run/"predictions_val.json"),checker.config)
        matches = sc.build_images(pred,labels,[i.stem for i in images],params)
        each = sc.evaluate_images(matches,params)
        evaluated[key] = {c:{g.img:g for g in entries} for c,entries in each.items()}
        groups["all140"] = list(range(140))
        scores[key] = {}
        for group, indexes in groups.items():
            select = np.zeros(len(images),dtype=bool)
            select[indexes] = True
            scores[key][group] = sc.accumulate(each,select,params)
        report = read(run/"run_report.json")
        assert abs(scores[key]["all140"]["map50_95"]-report["val_estimate"]["map50_95"]) < 1e-9
        cfg = yaml.safe_load((run/"runs/train/args.yaml").read_text())
        for k in ("data","project","save_dir"):
            cfg.pop(k,None)
        configs.append(cfg)
    differences = {k: [configs[0].get(k), configs[1].get(k)] for k in configs[0].keys() | configs[1].keys()
                   if configs[0].get(k) != configs[1].get(k)}
    allowed = {"epochs", "patience"} if args.allow_schedule_change else set()
    training_changes = {k: v for k, v in differences.items() if k not in allowed}
    if training_changes != {k: list(v) for k, v in CHANGES[args.experiment].items()}:
        raise ValueError(f"Unexpected training changes: {differences}")
    if args.experiment != "control":
        control, trial = (read(p / "resources.json") for p in (args.baseline, args.pilot))
        assert control.get("experiment", "control") == "control" and trial["experiment"] == args.experiment
        for k in ("dataset_sha256", "initial_weights_sha256", "notebook_sha256"):
            assert control[k] == trial[k]
        if args.experiment == "repeat-rare":
            sampling = read(args.pilot / "sampling.json")
            assert sampling["unique_train"] < sampling["draws_per_epoch"] <= 2 * sampling["unique_train"]
    rng = np.random.default_rng(26)
    diffs = []
    for _ in range(300):
        sample = []
        for group in scenes.values():
            clusters = list(group.values())
            for i in rng.integers(0,len(clusters),size=len(clusters)):
                sample.extend(clusters[i])
        sample.sort()
        vals = []
        for by_class in evaluated.values():
            per = {c:[entries[i] for i in sample if i in entries] for c,entries in by_class.items()}
            vals.append(sc.accumulate(per,np.ones(140,dtype=bool),params)["map50_95"])
        diffs.append(100*(vals[1]-vals[0]))
    n = 0
    tiny = defaultdict(int)
    for split in ("train","val"):
        for key,raw in read(args.baseline/"dataset/source_map.json")["files"].items():
            if not key.startswith(split+"/"):
                continue
            img = args.baseline/"raw"/raw
            with Image.open(img) as im:
                w,h = im.size
            for c,poly in sc.read_yolo_seg_labels(args.baseline/"raw/labels"/split/(img.stem+".txt"),11):
                area = polygon_area_at_640(poly,w,h)
                _,_,ref = sc.labels_to_gt([(c,poly)],h,w)
                assert area == ref[0]
                n += 1
                tiny[split] += int(area < 64)
    path = args.pilot/"submission/submission.zip"
    with ZipFile(path) as z:
        assert z.namelist() == ["model.onnx"] and path.stat().st_size<=42*2**20
        data = z.read("model.onnx")
    graph = onnx.load_model_from_string(data)
    onnx.checker.check_model(graph)
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2
    session=ort.InferenceSession(data,sess_options=opts,providers=["CPUExecutionProvider"])
    pt=args.pilot/"runs/train/weights/best.pt"
    net=YOLO(str(pt)).model.cpu().float().eval().fuse(verbose=False)
    head=net.model[-1]
    head.export=True
    head.format="onnx"
    head.dynamic=False
    torch.set_num_threads(2)
    parity=[]
    for img in (images[0],images[45],images[100]):
        with Image.open(img) as original:
            im=original.convert("RGB")
        im.thumbnail((640,640))
        canvas=Image.new("RGB",(640,640),(114,114,114))
        canvas.paste(im,((640-im.width)//2,(640-im.height)//2))
        x=np.asarray(canvas).transpose(2,0,1)[None].astype(np.float32)/255
        # Use native CPU convolutions for the FP32 reference. Record oneDNN
        # differences separately; kernel fusion changes the accumulation order.
        tensor = torch.from_numpy(x.copy())
        with torch.inference_mode():
            torch.backends.mkldnn.enabled = True
            optimized = net(tensor)
            torch.backends.mkldnn.enabled = False
            outputs = net(tensor)
        reference=session.run(None,{session.get_inputs()[0].name:x})
        for a,b,fast in zip(outputs,reference,optimized,strict=True):
            np.testing.assert_allclose(a.numpy(),b,atol=.003,rtol=.0001)
            fast_delta = np.abs(fast.numpy() - b)
            parity.append(dict(file=img.name,shape=list(b.shape),max_abs=float(np.abs(a.numpy()-b).max()),
                               torch_reference="CPU native FP32", onednn_max_abs=float(fast_delta.max()),
                               onednn_outside_tolerance=int((fast_delta > .003 + .0001*np.abs(b)).sum())))
    value=dict(scores=scores, paired_scene_bootstrap_95_points=np.percentile(diffs,[2.5,97.5]).tolist(),
               bootstrap_replicates=len(diffs), area_comparisons=n,tiny_by_split=tiny,
               validation_sha_verified=len(manifest),training_config_equal=not differences,
               training_config_differences=differences, experiment=args.experiment, onnx_parity=parity,
               opsets={i.domain:i.version for i in graph.opset_import},onnx_sha256=sha(data),
               best_sha256=sha(pt.read_bytes()),zip_sha256=sha(path.read_bytes()),zip_bytes=path.stat().st_size)
    args.out.write_text(json.dumps(value,indent=1))
    print({g:[round(100*scores[k][g]["map50_95"],2) for k in scores] for g in scores["pilot"]})
    print(n,dict(tiny),value["paired_scene_bootstrap_95_points"])


if __name__=="__main__":
    main()
