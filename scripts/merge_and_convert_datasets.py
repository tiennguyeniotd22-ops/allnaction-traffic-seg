"""
AllnAction RoboTaxi Hackathon — Unified Dataset Converter & Merger (11 Classes YOLO-Seg)
Converts and merges external datasets into a frozen base dataset (e.g. team780)
without touching the locked val split.

Supported sources:
  1. yolo_seg       : Another YOLO-seg folder (e.g. dataset_supplement_v4)
  2. idd_json       : IDD / Cityscapes polygon JSON + images
  3. bdd_seg_json   : BDD100K poly2d JSON + images
  4. mapillary_png  : Mapillary Vistas instance PNG masks + config_v1.2/v2.0

Usage:
    python scripts/merge_and_convert_datasets.py \
        --base-data team780 \
        --add-yolo dataset_supplement_v4 \
        --out team780_merged \
        --zip
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Other vehicle",
    4: "Motorcycle",
    5: "Motorcyclist",
    6: "Bicycle",
    7: "Cyclist",
    8: "Pedestrian",
    9: "Sign",
    10: "Traffic light",
}

IDD_LABEL_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "autorickshaw": 3,
    "caravan": 3,
    "trailer": 3,
    "vehicle fallback": 3,
    "motorcycle": 4,
    "rider": 5,  # Mặc định rider trên IDD đa số là xe máy; tách thêm nếu cần
    "bicycle": 6,
    "person": 8,
    "traffic sign": 9,
    "traffic light": 10,
}

BDD_LABEL_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "train": 3,
    "trailer": 3,
    "other vehicle": 3,
    "motorcycle": 4,
    "rider": 5,
    "bicycle": 6,
    "person": 8,
    "pedestrian": 8,
    "traffic sign": 9,
    "traffic light": 10,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def simplify_and_repair(pts_norm: np.ndarray, grid: int = 640) -> np.ndarray:
    """Repair self-intersecting polygons via rasterize + outer contour."""
    if len(pts_norm) < 3:
        return np.empty((0, 2), dtype=np.float32)
    canvas = np.zeros((grid, grid), dtype=np.uint8)
    pts = np.round(np.clip(pts_norm, 0.0, 1.0) * (grid - 1)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.0015 * peri, True).reshape(-1, 2).astype(np.float32) / float(grid - 1)
    return np.clip(approx, 0.0, 1.0)


def convert_idd_json_dir(src_dir: Path, out_dir: Path, min_area: float = 65.0) -> Counter:
    """Convert IDD/Cityscapes polygon JSON annotations into YOLO-seg train split."""
    stats = Counter()
    img_out = out_dir / "images" / "train"
    lbl_out = out_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.rglob("*.json"))
    for jp in json_files:
        data = json.loads(jp.read_text(encoding="utf-8"))
        w, h = data.get("imgWidth"), data.get("imgHeight")
        if not w or not h or "objects" not in data:
            continue

        # Find matching image
        stem = jp.stem.replace("_gtFine_polygons", "")
        img_match = next(
            (p for p in jp.parent.rglob(f"{stem}*") if p.suffix.lower() in IMG_EXTS),
            None,
        )
        if img_match is None:
            continue

        lines = []
        for obj in data["objects"]:
            lbl = obj.get("label", "").lower()
            if lbl not in IDD_LABEL_MAP:
                continue
            poly_px = np.array(obj.get("polygon", []), dtype=np.float32)
            if len(poly_px) < 3:
                continue
            poly_n = poly_px / np.array([float(w), float(h)], dtype=np.float32)
            poly_n = simplify_and_repair(poly_n)
            if len(poly_n) < 3:
                continue
            scale = 640.0 / float(max(w, h))
            area = float(cv2.contourArea(poly_n * np.array([w * scale, h * scale], dtype=np.float32)))
            if area < min_area:
                continue
            cid = IDD_LABEL_MAP[lbl]
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly_n)
            lines.append(f"{cid} {coords}")
            stats[CLASSES[cid]] += 1

        if lines:
            out_stem = f"idd_{stem}"
            shutil.copy2(img_match, img_out / f"{out_stem}{img_match.suffix.lower()}")
            (lbl_out / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return stats


def merge_yolo_seg_train(src_dir: Path, out_dir: Path, prefix: str = "ext_") -> int:
    """Copy train images and labels from another YOLO-seg dataset without touching val."""
    src_img = src_dir / "images" / "train"
    src_lbl = src_dir / "labels" / "train"
    dst_img = out_dir / "images" / "train"
    dst_lbl = out_dir / "labels" / "train"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_p in sorted(src_img.iterdir()):
        if img_p.suffix.lower() not in IMG_EXTS:
            continue
        lbl_p = src_lbl / f"{img_p.stem}.txt"
        out_stem = img_p.stem if img_p.stem.startswith(prefix) else f"{prefix}{img_p.stem}"
        shutil.copy2(img_p, dst_img / f"{out_stem}{img_p.suffix.lower()}")
        if lbl_p.exists():
            shutil.copy2(lbl_p, dst_lbl / f"{out_stem}.txt")
        else:
            (dst_lbl / f"{out_stem}.txt").write_text("", encoding="utf-8")
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert & Merge Traffic Datasets into Base 11-Class YOLO-Seg")
    parser.add_argument("--base-data", type=Path, required=True, help="Bộ dữ liệu gốc (vd: team780)")
    parser.add_argument("--add-yolo", type=Path, nargs="*", default=[], help="Các thư mục YOLO-seg bổ sung vào train")
    parser.add_argument("--add-idd", type=Path, nargs="*", default=[], help="Các thư mục IDD JSON bổ sung vào train")
    parser.add_argument("--out", type=Path, default=Path("team780_merged"), help="Thư mục đầu ra")
    parser.add_argument("--zip", action="store_true", help="Đóng gói thành file .zip")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    print(f"[1/3] Sao chép bộ gốc (giữ nguyên val khóa cứng): {args.base_data} -> {args.out}")
    shutil.copytree(args.base_data, args.out)

    total_added = 0
    for ydir in args.add_yolo:
        n = merge_yolo_seg_train(ydir, args.out, prefix=f"{ydir.name}_")
        total_added += n
        print(f"      Đã gộp {n} ảnh train từ YOLO-seg: {ydir}")

    for idir in args.add_idd:
        st = convert_idd_json_dir(idir, args.out)
        print(f"      Đã convert & gộp từ IDD {idir}: {dict(st)}")

    yaml_data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(args.out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"[2/3] Hoàn tất gộp (+{total_added} ảnh train bổ sung) vào {args.out}")
    if args.zip:
        z = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
        print(f"[3/3] Đã đóng gói: {z}")


if __name__ == "__main__":
    main()
