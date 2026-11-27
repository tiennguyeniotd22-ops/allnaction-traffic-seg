"""Reconstruct Team 21's manually reviewed changes for the fixed 50-scene audit."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from shapely import make_valid
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from overnight_apply_vn_review import file_sha
from overnight_coco_rider_edits import reviewed_polygon
from overnight_dataset import read, sha
from overnight_drafts import outline
from prepare_source_pilot import COCO_SHA256


def decide(idx, lines, size, native):
    actions = []

    def row(index, op, reason, **kw):
        original = lines[index]
        actions.append(dict(op=op, row=index, from_cls=int(original.split()[0]),
                            old_row_sha256=sha(original.encode()), before=original, reason=reason, **kw))

    def geom(index):
        return make_valid(Polygon(np.array(lines[index].split()[1:], float).reshape(-1, 2) * size))

    def add(cls, pixels, reason):
        actions.append(dict(op="append", cls=cls, pixels=pixels, reason=reason))

    def replace(index, cls, pixels, reason):
        row(index, "replace", reason, cls=cls, pixels=pixels)

    def from_native(index, cls, ids, reason):
        item = reviewed_polygon(native, ids, [], cls, *size)
        if index is None:
            item["reason"] = reason
            actions.append(item)
        else:
            item.pop("op")
            row(index, "replace", reason, **item)

    if idx == 1:
        reason = "§3 Cyclist: child is seated with adult on the same bicycle; both wheels belong to this instance."
        from_native(3, 7, [256597, 203711, 1768774], reason)
        row(7, "delete", reason)
        # The source crops the white helmet off the foreground adult. Trace its visible top.
        helmet = Polygon([(247,95),(248,83),(254,73),(263,67),(276,64),(288,65),(298,71),
                          (304,83),(307,101),(295,94),(283,90),(266,90),(253,96)])
        from_native(2, 7, [183301, 1765964], "§2.5 & §3 Cyclist: recover visible wheels, adult and worn helmet.")
        a = actions[-1]
        a["pixels"] = outline(make_valid(Polygon(a["pixels"])).union(helmet), *size).tolist()
        a["source_iou_before_manual_helmet"] = a.pop("source_iou")
    elif idx == 2:
        for pts in [
            [(163,157),(182,156),(183,176),(164,177)],
            [(185,155),(210,155),(211,180),(185,180)],
            [(218,158),(269,157),(270,177),(218,178)],
            [(273,158),(294,158),(295,181),(274,182)],
            [(358,297),(375,280),(393,296),(376,314)],
        ]:
            add(9, pts, "§3 Sign: independently visible traffic sign face on the gantry or right verge.")
    elif idx == 4:
        from_native(24, 4, [153747], "§3 Motorcycle: parked motorcycle, no one on its seat.")
        from_native(None, 8, [199294], "§3 Pedestrian: black-shirted man stands behind the row of parked motorcycles.")
    elif idx == 5:
        from_native(4, 5, [148639, 188895, 1249128], "§3 Motorcyclist: driver, passenger and motorcycle form one instance.")
        row(3, "delete", "§3 Motorcyclist: passenger/driver already included in the restored group.")
    elif idx == 6:
        from_native(7, 7, [1679644, 1765057], "§3 Cyclist: foreground helmeted person rides the cropped bicycle.")
        row(13, "delete", "§3 Cyclist: front bicycle merged with rider.")
    elif idx == 8:
        for pts in [[(456,143),(497,145),(497,154),(456,153)],
                    [(457,159),(497,160),(496,170),(457,168)],
                    [(457,175),(496,175),(496,185),(458,184)]]:
            add(9, pts, "§3 Sign: separate directional sign face on the signpost above the brasserie.")
    elif idx == 9:
        row(10, "delete", "§2.1: same full-sized white coach as Bus row 11, duplicated by source as Truck.")
    elif idx == 21:
        row(5, "delete", "§2.3 & §3 Sign: tiny amber strip on truck roof; visible height 9 px in 1280 px frame.")
    elif idx == 24:
        add(9, [(96,52),(350,236),(354,242),(359,309),(351,312),(90,162),(87,155)],
            "§3 Sign: official Hollywood street-name panel, visible face only.")
        row(2, "relabel", "§3 Car: passenger van with side windows.", cls=0)
    elif idx in (25, 38, 45):
        for j in ({25:[3], 38:[4], 45:[3,4]}[idx]):
            row(j, "delete", "§2.7: occupant seen through vehicle glass; vehicle already covers this patch.")
    elif idx == 26:
        row(2, "relabel", "§3 Other vehicle: motorized three-wheeler, two wheels on the wide rear axle.", cls=3)
        # Keep the truck pixels to the right of the foreground metal fork.
        visible = geom(1).intersection(Polygon([(427,138),(600,138),(600,246),(467,246),(448,202),(427,153)]))
        replace(1, 2, outline(visible,*size).tolist(), "§2.2: front fork occludes the background vehicle.")
    elif idx == 28:
        row(1, "delete", "§2.3: bus-like patch on distant bridge has visible box height 5 px at 640; ambiguous.")
    elif idx == 30:
        row(19, "delete", "§2.1 & §3 Traffic light: narrow duplicate inside the same signal box in row 2.")
    elif idx == 33:
        replace(0, 9, [(260,433),(326,433),(326,551),(260,551)], "§3 Sign: include the white rectangular backing of the lower prohibition sign.")
        replace(1, 9, [(259,239),(326,239),(326,356),(260,356)], "§3 Sign: include the white rectangular backing of the upper prohibition sign.")
    elif idx == 46:
        row(2, "delete", "§3 Traffic light: glowing street lamp, no traffic-signal housing.")
        replace(0, 9, [(137,504),(156,483),(182,484),(200,506),(199,538),(190,549),
                       (151,547),(137,537)], "§2.2 & §3 Sign: STOP face ends at the roof of the foreground car.")
        add(9, [(250,531),(275,530),(274,582),(250,572)], "§3 Sign: visible part of speed limit sign behind the vehicle.")
    elif idx == 48:
        # Solid crane body and telescopic boom, with individually traced visible lattice bars.
        solid = Polygon([(111,225),(110,202),(112,182),(126,179),(125,169),(118,169),(119,163),
                         (128,163),(128,158),(145,159),(148,166),(153,167),(161,121),(177,120),
                         (197,0),(246,0),(243,9),(236,9),(229,34),(225,34),(217,155),
                         (235,158),(267,156),(283,160),(289,177),(293,186),(244,203),(202,214),
                         (150,232),(137,233)])
        bars = [[(166,0),(135,106),(145,120),(170,119),(200,0)],
                [(181,0),(147,117)],[(144,78),(180,74),(151,102),(170,103),(145,117)],
                [(152,52),(186,48),(154,66),(181,66)],[(161,26),(193,24),(162,43),(188,42)],
                [(167,4),(198,3),(168,21),(191,19)]]
        crane = unary_union([solid]+[LineString(b).buffer(1.7) for b in bars])
        foreground = unary_union([geom(j) for j in range(len(lines))])
        add(3, outline(crane.difference(foreground), *size).tolist(), "§3 Other vehicle: exposed body, cabin, boom and lattice of crane behind trucks.")
    elif idx == 49:
        replace(7, 9, [(619,169),(649,168),(650,212),(640,211),(622,201)],
                "§3 Sign: individual left green highway panel; wet glass blurs the left lower edge.")
        add(9, [(652,168),(687,165),(697,166),(696,213),(652,212)], "§3 Sign: separate adjacent right highway panel.")
    return actions


def main():
    p = argparse.ArgumentParser()
    for key in ("dataset", "sample", "coco_zip", "native_manifest", "out"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    sample = read(a.sample)
    assert file_sha(a.dataset) == sample["dataset_sha256"] and len(sample["records"]) == 50
    with ZipFile(a.coco_zip) as z:
        body = z.read("annotations/instances_val2017.json")
    assert sha(body) == COCO_SHA256
    original = {b["id"]: b for b in json.loads(body)["annotations"]}
    parents = {r["key"]: r for r in read(a.native_manifest)}
    records, ledger = [], []
    unresolved = {
        4:"Further overlapping parked scooters have uncertain boundaries.",
        12:"Two riders overlap; their full silhouettes need a closer adjudication.",
        13:"Cargo cart in the middle distance: propulsion uncertain.",
        20:"Two distant car outlines overlap in the dark.",
        21:"Adjacent signal outlines blurred by rain.",
        22:"Background truck/car and adjacent signal boxes need adjudication.",
        23:"Two background car outlines overlap.",
        26:"Fine spokes, fork gaps and background vehicle type remain uncertain.",
        28:"Far bridge traffic remains below or near the size threshold.",
        36:"Separate bus-stop timetable below sign: official status unresolved.",
        41:"Far green street sign has a short visible side near the threshold.",
    }
    with ZipFile(a.dataset) as z:
        for r in sample["records"]:
            idx, name = r["index"], r["image"]
            stem = Path(name).stem
            size = r["size"]
            label_name = "labels/train/" + stem + ".txt"
            old = z.read(label_name)
            assert sha(old) == r["label_sha256"] and sha(z.read(name)) == r["image_sha256"]
            lines = old.decode().splitlines()
            local = decide(idx, lines, size, original)
            siblings = [name]
            source = "BDD100K" if stem.startswith("bdd_") else "Vietnam traffic"
            meta = {}
            if stem.startswith("zz_drive_"):
                source = "COCO val2017"
                suffix = stem.removeprefix("zz_drive_") + ".jpg"
                siblings = [n for n in z.namelist() if n.startswith("images/") and n.endswith(suffix)]
                assert all(n.startswith("images/train/zz_drive_") for n in siblings)
                meta["coco_image_id"] = int(Path(suffix).stem)
                assert all(original[j]["image_id"] == meta["coco_image_id"] for op in local for j in op.get("source_ids", []))
            elif stem.startswith("zz_plus_cc"):
                m = parents[stem.removeprefix("zz_plus_")]
                assert m["image_sha256"] == r["image_sha256"]
                source = m["source"]
                meta.update(coco_image_id=int(stem.split("cc")[-1]), url=m["url"], license=m["license"])
            elif stem.startswith("zz_plus_kt"):
                source = "KITTI"
            elif stem.startswith(("zz_new_", "zz_plus_")):
                source = "See original dataset provenance manifest"
            ledger.append(dict(**r, source=source, status="edited" if local else
                               ("unresolved" if idx in unresolved else "no_confirmed_change"),
                               remaining_question=unresolved.get(idx), n_actions=len(local)))
            if not local:
                continue
            for image in siblings:
                raw = z.read(image)
                assert list(Image.open(BytesIO(raw)).size) == size
                data = z.read("labels/train/" + Path(image).stem + ".txt")
                target = data.decode().splitlines()
                assert len(target) == len(set(target))
                # Reordered and previously QC'd sibling files can differ on unrelated rows.
                actions = [dict(op) for op in local]
                for op in actions:
                    if "row" in op:
                        op["row"] = target.index(op["before"])
                records.append(dict(image=image, size=size, image_sha256=sha(raw), label_sha256=sha(data),
                                    index=idx, source=source, **meta, actions=actions))
    report = dict(base_sha256=sample["dataset_sha256"], sample_sha256=file_sha(a.sample),
                  source_json_sha256=COCO_SHA256, records=records, reviewed_scenes=ledger)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(len(records), Counter(a["op"] for r in records for a in r["actions"]))


if __name__ == "__main__":
    main()
