"""
AllnAction RoboTaxi Hackathon — Auto-Prelabel Pipeline (YOLO Segmentation 1.0)
Official 11 Classes (0..10) + Smart Rider Merge (Person + Motorcycle -> Motorcyclist).

Usage on Colab / GPU Server:
    pip install -U ultralytics opencv-python numpy pyyaml
    python scripts/prelabel_yolo_seg.py --images raw_images/ --out prelabeled_data --model yolo26x-seg.pt --zip
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

if __package__:
    from .mask_geometry import export_coordinates, polygon_area_at_640
else:
    from mask_geometry import export_coordinates, polygon_area_at_640

# ==============================================================================
# 1. BẢNG 11 LỚP CHÍNH THỨC CỦA CUỘC THI
# ==============================================================================
COMPETITION_CLASSES = {
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

# COCO IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light, 11=stop sign
COCO_TO_COMPETITION = {
    2: 0,    # car -> Car
    5: 1,    # bus -> Bus
    7: 2,    # truck -> Truck
    3: 4,    # motorcycle -> Motorcycle (không người)
    1: 6,    # bicycle -> Bicycle (không người)
    0: 8,    # person -> Pedestrian (sẽ tự gộp thành Motorcyclist/Cyclist nếu đi kèm xe)
    11: 9,   # stop sign -> Sign
    9: 10,   # traffic light -> Traffic light
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def simplify_polygon(poly_norm: np.ndarray, epsilon_ratio: float = 0.002) -> np.ndarray:
    """Simplify normalized polygon vertices while keeping >= 3 points."""
    if len(poly_norm) < 3:
        return poly_norm
    pts = (poly_norm * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, epsilon_ratio * peri, True).reshape(-1, 2) / 1000.0
    if len(approx) >= 3:
        return np.clip(approx, 0.0, 1.0)
    return np.clip(poly_norm, 0.0, 1.0)


def box_iou_or_rider_overlap(box_person: np.ndarray, box_bike: np.ndarray) -> bool:
    """Check if a person bbox overlaps/rides on top of a motorcycle/bicycle bbox."""
    px1, py1, px2, py2 = box_person
    bx1, by1, bx2, by2 = box_bike

    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return False

    bike_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    person_area = max(1e-6, (px2 - px1) * (py2 - py1))
    # Người lái thường giao >= 15% diện tích xe hoặc tâm ngang nằm trong vùng xe
    p_cx = 0.5 * (px1 + px2)
    return (inter / min(bike_area, person_area) >= 0.15) and (bx1 - 0.05 <= p_cx <= bx2 + 0.05)


def merge_two_polygons(poly_a: np.ndarray, poly_b: np.ndarray, grid: int = 640) -> np.ndarray:
    """Rasterize and extract outer contour of two merged polygons (person + 2-wheeler)."""
    canvas = np.zeros((grid, grid), dtype=np.uint8)
    for poly in (poly_a, poly_b):
        pts = np.round(poly * grid).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], 255)
    # Nối nhẹ khoảng hở nhỏ giữa người và yên xe nếu có
    kernel = np.ones((5, 5), np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly_b
    largest = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) / float(grid)
    return simplify_polygon(largest)


def export_cvat_yaml(out_dir: Path) -> None:
    """Write data.yaml compatible with CVAT YOLO Segmentation 1.0 & Ultralytics."""
    yaml_data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(COMPETITION_CLASSES),
        "names": COMPETITION_CLASSES,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)


def run_prelabel(
    images_dir: Path,
    out_dir: Path,
    model_weights: str,
    conf: float,
    iou: float,
    imgsz: int,
    min_area_640: float,
    split: str,
    make_zip: bool,
) -> None:
    """Run YOLO-seg inference, merge riders, and export clean YOLO Segmentation labels."""
    from ultralytics import YOLO

    image_paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {images_dir}")

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    export_cvat_yaml(out_dir)

    print(f"[1/3] Nạp model prelabel: {model_weights} | Ảnh: {len(image_paths)}")
    model = YOLO(model_weights)

    stats = Counter()
    dropped_small = 0
    merged_riders = 0

    print(f"[2/3] Đang chạy prelabel + tách Motorcyclist/Cyclist (conf={conf}, imgsz={imgsz})...")
    for idx, img_path in enumerate(image_paths, start=1):
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            retina_masks=True,
            verbose=False,
        )
        res = results[0]
        height, width = res.orig_shape
        lines = []

        if res.masks is not None and res.boxes is not None:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            boxes_n = res.boxes.xyxyn.cpu().numpy()
            polys = res.masks.xyn

            persons = [
                (i, boxes_n[i], polys[i])
                for i, c in enumerate(cls_ids)
                if c == 0 and len(polys[i]) >= 3
            ]
            used_persons = set()
            final_objects: list[tuple[int, np.ndarray]] = []

            # 1) Kiểm tra Motorcycle (3) & Bicycle (1) để gộp thành Motorcyclist (5) / Cyclist (7)
            for i, coco_cls in enumerate(cls_ids):
                if coco_cls not in (1, 3) or len(polys[i]) < 3:
                    continue
                matched_person = None
                for p_idx, p_box, p_poly in persons:
                    if p_idx in used_persons:
                        continue
                    if box_iou_or_rider_overlap(p_box, boxes_n[i]):
                        matched_person = (p_idx, p_poly)
                        break

                if matched_person is not None:
                    p_idx, p_poly = matched_person
                    used_persons.add(p_idx)
                    merged_poly = merge_two_polygons(p_poly, polys[i])
                    target_cls = 5 if coco_cls == 3 else 7  # Motorcyclist or Cyclist
                    final_objects.append((target_cls, merged_poly))
                    merged_riders += 1
                else:
                    target_cls = 4 if coco_cls == 3 else 6  # Motorcycle or Bicycle (không người)
                    final_objects.append((target_cls, polys[i]))

            # 2) Các lớp còn lại (bỏ qua person đã gộp vào xe)
            for i, coco_cls in enumerate(cls_ids):
                if coco_cls in (1, 3) or i in used_persons:
                    continue
                if coco_cls not in COCO_TO_COMPETITION or len(polys[i]) < 3:
                    continue
                final_objects.append((COCO_TO_COMPETITION[coco_cls], polys[i]))

            # 3) Lọc polygon hợp lệ & >= 64 px ở 640x640
            for comp_cls, poly in final_objects:
                poly_clean = simplify_polygon(poly)
                if len(poly_clean) < 3:
                    continue
                coords_str, exported = export_coordinates(poly_clean)
                if polygon_area_at_640(exported, width, height) < min_area_640:
                    dropped_small += 1
                    continue
                lines.append(f"{comp_cls} {coords_str}")
                stats[COMPETITION_CLASSES[comp_cls]] += 1

        shutil.copy2(img_path, img_out / img_path.name)
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"   Đã xử lý {idx}/{len(image_paths)} ảnh...")

    summary = {
        "total_images": len(image_paths),
        "total_instances": sum(stats.values()),
        "auto_merged_riders": merged_riders,
        "dropped_small_masks": dropped_small,
        "instances_per_class": dict(stats),
    }
    (out_dir / "prelabel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[3/3] === TỔNG KẾT PRELABEL ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if make_zip:
        zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"Đã đóng gói sẵn sàng import CVAT / train: {zip_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-Prelabel YOLO Segmentation 1.0 (11 Classes)")
    parser.add_argument("--images", type=Path, required=True, help="Thư mục chứa ảnh thô")
    parser.add_argument("--out", type=Path, default=Path("prelabeled_data"), help="Thư mục xuất")
    parser.add_argument("--model", type=str, default="yolo26x-seg.pt", help="Model pretrain để prelabel")
    parser.add_argument("--conf", type=float, default=0.30, help="Ngưỡng confidence")
    parser.add_argument("--iou", type=float, default=0.65, help="Ngưỡng NMS IoU")
    parser.add_argument("--imgsz", type=int, default=960, help="Kích thước ảnh khi prelabel")
    parser.add_argument("--min-area", type=float, default=64.0, help="Diện tích mask tối thiểu ở 640x640")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split xuất ra")
    parser.add_argument("--zip", action="store_true", help="Đóng gói thành file .zip")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prelabel(
        images_dir=args.images,
        out_dir=args.out,
        model_weights=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        min_area_640=args.min_area,
        split=args.split,
        make_zip=args.zip,
    )
