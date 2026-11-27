"""Apply COCO review decisions to train candidates; retain source IDs and before/after hashes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from overnight_dataset import encode, read, sha
from overnight_drafts import outline
from overnight_extend import check_labels


def write(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Use a fresh output directory for changed content: {path}")
    path.write_bytes(content)


def review(root, decisions, out):
    sources = {}
    for name, expected in decisions["inputs"].items():
        source = root / name / "manifest.json"
        assert sha(source.read_bytes()) == expected
        sources[name] = {r["key"]: r for r in read(source)}
    for sub in ["images", "labels", "views"]:
        (out / sub).mkdir(exist_ok=True, parents=True)
    records = []
    for key, decision in sorted(decisions["keep"].items()):
        name = decision["folder"]
        original = sources[name][key]
        w, h = original["size"]
        assert original["license"]["id"] in (4, 5)
        image = (root / name / "images" / (key + ".jpg")).read_bytes()
        before = (root / name / "labels" / (key + ".txt")).read_bytes()
        assert sha(before) == original["label_sha256"]
        assert sha(image) == original["image_sha256"] == original["parent_sha256"]
        rows, shapes, classes = {}, {}, {}
        for i, (obj, line) in enumerate(zip(original["objects"], before.decode().splitlines(), strict=True)):
            if i in decision.get("drop", []):
                continue
            tokens = line.split()
            classes[i] = decision.get("relabel", {}).get(str(i), int(tokens[0]))
            shapes[i] = Polygon(np.asarray(tokens[1:], float).reshape(-1, 2) * [w, h])
            if str(i) in decision.get("union", {}):
                shapes[i] = shapes[i].union(Polygon(decision["union"][str(i)]))
            rows[i] = {"native_ids": obj["native_ids"], "source_class": int(tokens[0]), "source_iou": obj["iou"]}
        for i, item in enumerate(decision.get("add", []), start=len(original["objects"])):
            classes[i] = item["cls"]
            shapes[i] = Polygon(item["xy"])
            rows[i] = {"native_ids": [], "method": "manual visible face"}
        front = decision.get("front", [])
        if front:
            assert set(front) == set(shapes) and len(front) == len(shapes)
            occupied = Polygon()
            for i in front:
                visible = shapes[i].difference(occupied)
                occupied = occupied.union(shapes[i])
                shapes[i] = visible
        lines, objects = [], []
        overlay = Image.open(root / name / "images" / (key + ".jpg")).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for i, shape in sorted(shapes.items()):
            edited = bool(front) or str(i) in decision.get("union", {})
            points = outline(shape, w, h) if edited else list(shape.exterior.coords)[:-1]
            line = encode(classes[i], points, w, h)
            xy = np.asarray(line.split()[1:], float).reshape(-1, 2) * [w, h]
            if min(np.ptp(xy, axis=0)) < max(w, h) * .0125:
                raise ValueError(f"{key}/{i} below the BTC review threshold")
            pts = [tuple(v) for v in xy]
            draw.line(pts + pts[:1], width=2, fill=["#ff5656", "#61ccff", "#eebd18", "#46cd87"][i % 4])
            x, y = min(pts, key=lambda t: t[1])
            draw.text((x, max(0, y-10)), str(classes[i]), fill="white", stroke_width=1, stroke_fill="black")
            lines.append(line)
            objects.append(dict(rows[i], cls=classes[i], pixels=round(Polygon(pts).area, 2)))
        label = ("\n".join(lines) + "\n").encode()
        check_labels(label, objects)
        write(out / "images" / (key + ".jpg"), image)
        write(out / "labels" / (key + ".txt"), label)
        overlay.save(out / "views" / (key + ".jpg"), quality=94)
        records.append(dict(original, source_label_sha256=sha(before), label_sha256=sha(label),
                            objects=objects, review=decision))
    write(out / "manifest.json", (json.dumps(records, indent=1) + "\n").encode())
    print(len(records), Counter(obj["cls"] for row in records for obj in row["objects"]))


def select(folder, scene_report, out, decisions):
    scenes = read(scene_report)
    manifest = folder / "manifest.json"
    assert sha(manifest.read_bytes()) == scenes["candidate_manifest_sha256"][folder.name]
    assert scenes["reference_sha256"]["probe-frozen.zip"]
    edges = [(e["a"], e["b"]) for e in scenes["same_scene"]] + [r[:2] for r in scenes["close_hash"]]
    keep, skipped = [], {}
    rejected = decisions.get("excluded_final_review", {})
    assert set(rejected).issubset(decisions["keep"])
    for row in read(manifest):
        key = row["key"]
        if key in rejected:
            continue
        conflicts = {b if a == key else a for a, b in edges if key in (a, b)}
        hits = sorted(conflicts.intersection({"base", *keep}))
        if hits:
            skipped[key] = hits
        else:
            keep.append(key)
    assert 0 < len(keep) <= 60
    write(out, (json.dumps(dict(keep=keep, excluded_overlap=skipped, excluded_final_review=rejected,
                               source_manifest_sha256=sha(manifest.read_bytes())), indent=1) + "\n").encode())
    print(len(keep), "retained after scene checks;", len(skipped), "excluded")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cmd = p.add_subparsers(dest="action", required=True)
    r = cmd.add_parser("review")
    for arg in ["root", "decisions", "out"]:
        r.add_argument(arg, type=Path)
    s = cmd.add_parser("select")
    for arg in ["folder", "scene_report", "out"]:
        s.add_argument(arg, type=Path)
    s.add_argument("--decisions", type=Path, required=True)
    a = p.parse_args()
    if a.action == "review":
        review(a.root, read(a.decisions), a.out)
    else:
        select(a.folder, a.scene_report, a.out, read(a.decisions))
