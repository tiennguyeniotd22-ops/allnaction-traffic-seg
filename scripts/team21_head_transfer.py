"""Warm-start four competition class rows missed by the pretrained name matcher."""
import json
import math

import torch
from ultralytics.models.yolo.segment import SegmentationTrainer


TARGETS = ("Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light")
SOURCES = ("car", "bus", "truck", None, "motorcycle", "motorcycle", "bicycle",
           "bicycle", "person", "stop sign", "traffic light")


@torch.no_grad()
def initialize_head(source, target):
    names = {v: k for k, v in source.names.items()}
    assert tuple(target.names[i] for i in range(11)) == TARGETS and len(names) == 80
    src, dst = source.model[-1], target.model[-1]
    assert type(src) is type(dst) and type(dst).__name__ == "Segment26"
    # The installed 8.4.214 loader preserves the pretrained hidden channels.
    pairs = [(src.proto.semseg[-1], dst.proto.semseg[-1])]
    for branch in ("cv3", "one2one_cv3"):
        pairs.extend((s[-1], t[-1]) for s, t in zip(getattr(src, branch), getattr(dst, branch), strict=True))
    assert len(pairs) == 7
    for original, new in pairs:
        assert original.weight.shape[1:] == new.weight.shape[1:]
        for i, name in enumerate(SOURCES):
            if name is None:
                continue
            j = names[name]
            if TARGETS[i].lower() == name:
                assert torch.equal(new.weight[i], original.weight[j])
            new.weight[i].copy_(original.weight[j])
            # Split the prior mass between the occupied and unoccupied variants.
            new.bias[i].copy_(original.bias[j] - math.log(SOURCES.count(name)))
    return dict(source_names=SOURCES, target_names=TARGETS, classifier_convolutions=len(pairs))


class TrafficTransferTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if weights is None:
            raise ValueError("This initialization requires the official COCO checkpoint")
        record = initialize_head(weights, model)
        (self.save_dir / "head-transfer.json").write_text(json.dumps(record, indent=1))
        print("Initialized 10/11 traffic class rows from COCO", flush=True)
        return model
