"""
AllnAction RoboTaxi Hackathon — Supplement Dataset Builder (COCO train2017 Rare Classes)
Extracts clean polygon instances from COCO train2017 (zero overlap with val2017)
specifically targeting the weak classes from Round 3:
  - 4: Motorcycle (empty), 6: Bicycle (empty), 7: Cyclist, 9: Sign, 10: Traffic light.

Usage:
    python scripts/build_supplement_data.py --out dataset_supplement_v4 --max-images 400 --zip
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from prepare_coco_traffic import (
    COCO_NAME_TO_COMP,
    COMPETITION_CLASSES,
    box_overlap_rider,
    coco_seg_to_norm_poly,
    download_annotations,
    download_one,
    merge_polygons,
    polygon_area_at_640,
    repair_and_simplify_polygon,
)

ANNOTATIONS_ZIP = Path("raw_cache/annotations_trainval2017.zip")


def extract_train_json(cache_dir: Path) -> Path:
    """Extract instances_train2017.json from cached COCO annotations zip (auto-downloads if missing)."""
    json_path = cache_dir / "annotations" / "instances_train2017.json"
    if json_path.exists():
        return json_path
    zip_path = cache_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download_annotations(cache_dir)
    print("[1/4] Đang giải nén instances_train2017.json (~469 MB)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("annotations/instances_train2017.json", cache_dir)
    return json_path


def build_supplement(
    out_dir: Path,
    cache_dir: Path,
    max_images: int,
    min_area_640: float,
    make_zip: bool,
) -> None:
    """Select, download, and convert top rare-class traffic images from COCO train2017."""
    json_path = extract_train_json(cache_dir)
    print(f"[2/4] Đang lọc ảnh hiếm từ {json_path.name}...")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_info = {im["id"]: im for im in data["images"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 0 and cat_id_to_name.get(ann["category_id"]) in COCO_NAME_TO_COMP:
            anns_by_img[ann["image_id"]].append(ann)

    # Bucketed selection so Motorcycle, Bicycle/Cyclist, and Sign/Traffic light all get boosted
    moto_imgs, bike_imgs, sign_imgs = [], [], []
    for img_id, anns in anns_by_img.items():
        names = [cat_id_to_name[a["category_id"]] for a in anns]
        c = Counter(names)
        if c["motorcycle"] >= 1:
            moto_imgs.append((c["motorcycle"] * 10 + len(anns), img_id))
        elif c["bicycle"] >= 1:
            bike_imgs.append((c["bicycle"] * 10 + c["person"] * 2, img_id))
        elif c["stop sign"] >= 1 or c["traffic light"] >= 2:
            sign_imgs.append((c["stop sign"] * 15 + c["traffic light"] * 5, img_id))

    moto_imgs.sort(reverse=True)
    bike_imgs.sort(reverse=True)
    sign_imgs.sort(reverse=True)

    n_moto = int(max_images * 0.40)
    n_bike = int(max_images * 0.40)
    n_sign = max_images - n_moto - n_bike

    selected_ids = (
        [i for _, i in moto_imgs[:n_moto]]
        + [i for _, i in bike_imgs[:n_bike]]
        + [i for _, i in sign_imgs[:n_sign]]
    )

    img_out = out_dir / "images" / "train"
    lbl_out = out_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Đang tải {len(selected_ids)} ảnh bổ sung (Motorcycle/Bicycle/Cyclist/Sign)...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [
            pool.submit(
                download_one,
                img_id_to_info[img_id]["coco_url"],
                img_out / f"coco_tr_{img_id_to_info[img_id]['file_name']}",
            )
            for img_id in selected_ids
        ]
        ok_count = sum(1 for fut in as_completed(futs) if fut.result())
    print(f"      Đã tải thành công {ok_count}/{len(selected_ids)} ảnh.")

    stats = Counter()
    dropped_small = 0
    for img_id in selected_ids:
        info = img_id_to_info[img_id]
        img_path = img_out / f"coco_tr_{info['file_name']}"
        if not img_path.exists():
            continue

        w, h = info["width"], info["height"]
        anns = anns_by_img[img_id]
        persons, two_wheelers, others = [], [], []

        for ann in anns:
            cname = cat_id_to_name[ann["category_id"]]
            poly = coco_seg_to_norm_poly(ann["segmentation"], w, h)
            if poly is None:
                continue
            if cname == "person":
                persons.append((ann["bbox"], poly))
            elif cname in ("motorcycle", "bicycle"):
                two_wheelers.append((cname, ann["bbox"], poly))
            else:
                others.append((COCO_NAME_TO_COMP[cname], poly))

        used_p = set()
        final_objs = []
        for cname, b_box, b_poly in two_wheelers:
            matched = None
            for p_idx, (p_box, p_poly) in enumerate(persons):
                if p_idx not in used_p and box_overlap_rider(p_box, b_box):
                    matched = (p_idx, p_poly)
                    break
            if matched is not None:
                p_idx, p_poly = matched
                used_p.add(p_idx)
                cls_id = 5 if cname == "motorcycle" else 7
                final_objs.append((cls_id, merge_polygons(p_poly, b_poly)))
            else:
                cls_id = 4 if cname == "motorcycle" else 6
                final_objs.append((cls_id, b_poly))

        for p_idx, (_, p_poly) in enumerate(persons):
            if p_idx not in used_p:
                final_objs.append((8, p_poly))
        final_objs.extend(others)

        lines = []
        for cls_id, poly in final_objs:
            poly = repair_and_simplify_polygon(poly)
            if len(poly) < 3 or polygon_area_at_640(poly, w, h) < min_area_640:
                dropped_small += 1
                continue
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls_id} {coords}")
            stats[COMPETITION_CLASSES[cls_id]] += 1

        lbl_path = lbl_out / f"coco_tr_{Path(info['file_name']).stem}.txt"
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

    yaml_data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(COMPETITION_CLASSES),
        "names": COMPETITION_CLASSES,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"\n[4/4] === THỐNG KÊ GÓI BỔ SUNG v4 (đã lọc {dropped_small} mask < {min_area_640}px) ===")
    print(dict(stats))

    if make_zip:
        zip_out = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"Đã đóng gói: {zip_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Rare-Class Supplement from COCO train2017")
    parser.add_argument("--out", type=Path, default=Path("dataset_supplement_v4"))
    parser.add_argument("--cache", type=Path, default=Path("raw_cache"))
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--min-area", type=float, default=65.0)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    build_supplement(
        out_dir=args.out,
        cache_dir=args.cache,
        max_images=args.max_images,
        min_area_640=args.min_area,
        make_zip=args.zip,
    )


if __name__ == "__main__":
    main()
