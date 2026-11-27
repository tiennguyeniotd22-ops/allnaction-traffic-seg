"""
AllnAction RoboTaxi Hackathon — Dataset Validator & Hard-Scene Augmentation
1. Validate YOLO-Seg dataset (polygon >= 3 pts, [0,1] bounds, area >= 64px, class counts).
2. Generate Hard-Scene Synthetic Data (Night / Low-Light + Rain) keeping polygon labels exact.

Usage:
    python scripts/validate_and_augment.py --data dataset_traffic_11 --augment-hard 0.3
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__:
    from .mask_geometry import polygon_area_at_640
else:
    from mask_geometry import polygon_area_at_640

CLASSES = [
    "Car", "Bus", "Truck", "Other vehicle", "Motorcycle",
    "Motorcyclist", "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_split(data_dir: Path, split: str, min_area_640: float = 64.0) -> dict:
    """Audit images and YOLO-Seg polygon labels for errors and class imbalance."""
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split
    if not img_dir.exists():
        return {"exists": False}

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    counts = Counter()
    bad_lines = 0
    small_masks = 0
    empty_labels = 0
    missing_labels = 0
    unreadable_images = 0

    for img_p in imgs:
        try:
            with Image.open(img_p) as image:
                width, height = image.size
        except OSError:
            unreadable_images += 1
            continue
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            missing_labels += 1
            continue
        text = lbl_p.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                bad_lines += 1
                continue
            try:
                cls_id = int(parts[0])
                coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
            except ValueError:
                bad_lines += 1
                continue
            if not (0 <= cls_id < len(CLASSES)) or not np.isfinite(coords).all() or np.any(coords < 0.0) or np.any(coords > 1.0):
                bad_lines += 1
                continue
            area = polygon_area_at_640(coords, width, height)
            if area < min_area_640:
                small_masks += 1
            counts[CLASSES[cls_id]] += 1

    return {
        "exists": True,
        "images": len(imgs),
        "empty_bg_images": empty_labels,
        "missing_labels": missing_labels,
        "unreadable_image_headers": unreadable_images,
        "bad_lines": bad_lines,
        "small_masks_lt_64px": small_masks,
        "class_counts": {c: counts[c] for c in CLASSES},
    }


def apply_night_effect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate night / low-light street scene with slight color shift and sensor noise."""
    gamma = rng.uniform(2.0, 2.8)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    dark = cv2.LUT(img, table)
    # Slight cool/amber streetlamp tint
    dark = dark.astype(np.float32)
    dark[:, :, 0] *= rng.uniform(0.85, 1.05)  # B
    dark[:, :, 2] *= rng.uniform(0.95, 1.15)  # R
    noise = rng.normal(0, 4.0, dark.shape).astype(np.float32)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def apply_rain_effect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate rain streaks + slight atmospheric haze."""
    h, w = img.shape[:2]
    rain_layer = np.zeros_like(img, dtype=np.uint8)
    n_drops = int(rng.integers(350, 700))
    slant = int(rng.integers(-8, 9))
    length = int(rng.integers(12, 24))

    for _ in range(n_drops):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        cv2.line(rain_layer, (x, y), (x + slant, y + length), (200, 210, 220), 1)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    hazy = cv2.addWeighted(img, 0.85, np.full_like(img, 180), 0.15, 0)
    return cv2.addWeighted(hazy, 1.0, rain_layer, 0.45, 0)


def augment_hard_scenes(data_dir: Path, ratio: float, seed: int = 2026) -> int:
    """Create night and rain synthetic copies in train/ with identical polygon labels."""
    rng = np.random.default_rng(seed)
    img_dir = data_dir / "images" / "train"
    lbl_dir = data_dir / "labels" / "train"
    if not img_dir.exists() or ratio <= 0.0:
        return 0

    base_imgs = [
        p for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS and not p.stem.startswith(("syn_night_", "syn_rain_"))
    ]
    n_aug = int(len(base_imgs) * ratio)
    chosen = rng.choice(base_imgs, size=min(n_aug, len(base_imgs)), replace=False)

    created = 0
    for img_p in chosen:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        mode = "night" if rng.random() < 0.5 else "rain"
        aug_img = apply_night_effect(img, rng) if mode == "night" else apply_rain_effect(img, rng)

        out_stem = f"syn_{mode}_{img_p.stem}"
        cv2.imwrite(str(img_dir / f"{out_stem}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{out_stem}.txt").write_text(lbl_p.read_text(encoding="utf-8"), encoding="utf-8")
        created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate & Augment Hard Traffic Scenes")
    parser.add_argument("--data", type=Path, required=True, help="Thư mục dataset YOLO-seg")
    parser.add_argument("--augment-hard", type=float, default=0.0, help="Tỷ lệ sinh thêm cảnh đêm/mưa vào train (vd: 0.3)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    for split in ("train", "val"):
        report = validate_split(args.data, split)
        print(f"\n=== VALIDATE [{split.upper()}] ===")
        print(report)

    if args.augment_hard > 0:
        n_created = augment_hard_scenes(args.data, args.augment_hard, args.seed)
        print(f"\n[AUGMENT] Đã sinh thêm {n_created} ảnh cảnh khó (đêm/mưa) vào {args.data}/images/train")


if __name__ == "__main__":
    main()
