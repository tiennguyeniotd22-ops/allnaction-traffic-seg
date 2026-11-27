"""
AllnAction RoboTaxi Hackathon — Offline Copy-Paste Simulator for Rare Classes
Extracts clean polygon instances of rare classes (3: Other vehicle, 4: Motorcycle,
6: Bicycle, 7: Cyclist) and blends them onto road regions of training images.

Usage:
    python scripts/copy_paste_rare.py \
        --src-data dataset_with_rare \
        --dst-data dataset_traffic_v2 \
        --classes 3 4 6 7 \
        --max-paste 200
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_seg_labels(lbl_path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse YOLO-seg .txt into list of (class_id, normalized_polygon Nx2)."""
    if not lbl_path.exists():
        return []
    items = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cls_id = int(float(parts[0]))
        poly = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        items.append((cls_id, np.clip(poly, 0.0, 1.0)))
    return items


def collect_object_bank(src_dir: Path, target_classes: set[int], min_px: int = 24) -> list[dict]:
    """Extract cropped object patches + binary masks + polygons for target classes."""
    bank = []
    img_dir = src_dir / "images" / "train"
    lbl_dir = src_dir / "labels" / "train"
    if not img_dir.exists():
        return bank

    for img_p in sorted(img_dir.iterdir()):
        if img_p.suffix.lower() not in IMG_EXTS:
            continue
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        objs = load_yolo_seg_labels(lbl_p)
        matching = [(c, poly) for c, poly in objs if c in target_classes]
        if not matching:
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        for cls_id, poly in matching:
            pts = np.round(poly * np.array([w, h], dtype=np.float32)).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            if bw < min_px or bh < min_px or bw > int(0.6 * w) or bh > int(0.6 * h):
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

            crop_img = img[y : y + bh, x : x + bw].copy()
            crop_mask = mask[y : y + bh, x : x + bw].copy()
            rel_poly = (pts - np.array([x, y], dtype=np.int32)).astype(np.float32)
            rel_poly[:, 0] /= float(bw)
            rel_poly[:, 1] /= float(bh)

            bank.append(
                {
                    "cls_id": cls_id,
                    "img": crop_img,
                    "mask": crop_mask,
                    "rel_poly": rel_poly,
                    "w": bw,
                    "h": bh,
                }
            )
    return bank


def paste_object_feathered(
    bg_img: np.ndarray,
    obj: dict,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Blend object onto lower road region of bg_img with alpha feathering."""
    H, W = bg_img.shape[:2]
    scale = rng.uniform(0.75, 1.15)
    nw = max(20, int(obj["w"] * scale))
    nh = max(20, int(obj["h"] * scale))
    if nw >= int(0.65 * W) or nh >= int(0.55 * H):
        return None

    patch_img = cv2.resize(obj["img"], (nw, nh), interpolation=cv2.INTER_LINEAR)
    patch_mask = cv2.resize(obj["mask"], (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Place in lower 60% of frame (road surface)
    min_y = max(0, int(0.42 * H))
    max_y = max(min_y + 1, H - nh)
    max_x = max(1, W - nw)
    px = rng.randint(0, max_x - 1)
    py = rng.randint(min_y, max_y - 1)

    roi = bg_img[py : py + nh, px : px + nw].astype(np.float32)

    # Match brightness lightly to background ROI
    fg_f = patch_img.astype(np.float32)
    fg_mean = fg_f[patch_mask > 127].mean() if np.any(patch_mask > 127) else 127.0
    bg_mean = roi.mean()
    gain = np.clip(bg_mean / max(fg_mean, 20.0), 0.70, 1.35)
    fg_f = np.clip(fg_f * gain, 0, 255)

    alpha = cv2.GaussianBlur(patch_mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None]
    blended = fg_f * alpha + roi * (1.0 - alpha)

    out_img = bg_img.copy()
    out_img[py : py + nh, px : px + nw] = np.clip(blended, 0, 255).astype(np.uint8)

    new_poly = obj["rel_poly"].copy()
    new_poly[:, 0] = (px + new_poly[:, 0] * nw) / float(W)
    new_poly[:, 1] = (py + new_poly[:, 1] * nh) / float(H)
    return out_img, np.clip(new_poly, 0.0, 1.0)


def run_copy_paste(
    src_data: Path,
    dst_data: Path,
    target_classes: set[int],
    max_paste: int,
    seed: int,
) -> None:
    """Create synthetic copy-paste training images in dst_data/images/train."""
    rng = random.Random(seed)
    bank = collect_object_bank(src_data, target_classes)
    if not bank:
        print(f"[CopyPaste] Không tìm thấy instance nào thuộc lớp {sorted(target_classes)} trong {src_data}")
        return

    print(f"[CopyPaste] Thu thập được {len(bank)} mask mẫu từ {src_data}")
    dst_img_dir = dst_data / "images" / "train"
    dst_lbl_dir = dst_data / "labels" / "train"
    dst_imgs = [
        p for p in sorted(dst_img_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS and not p.stem.startswith("cp_sim_")
    ]

    created = 0
    for idx in range(max_paste):
        bg_path = rng.choice(dst_imgs)
        bg_lbl_path = dst_lbl_dir / f"{bg_path.stem}.txt"
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue

        existing = load_yolo_seg_labels(bg_lbl_path)
        n_objs = rng.randint(1, 2)
        cur_img = bg_img
        added = []
        for _ in range(n_objs):
            obj = rng.choice(bank)
            res = paste_object_feathered(cur_img, obj, rng)
            if res is not None:
                cur_img, new_poly = res
                added.append((obj["cls_id"], new_poly))

        if not added:
            continue

        all_objs = existing + added
        lines = []
        for cls_id, poly in all_objs:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls_id} {coords}")

        out_stem = f"cp_sim_{idx:04d}_{bg_path.stem}"
        cv2.imwrite(str(dst_img_dir / f"{out_stem}.jpg"), cur_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (dst_lbl_dir / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        created += 1

    print(f"[CopyPaste] Đã tạo {created} ảnh simulation vào {dst_img_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Copy-Paste Simulator for Rare Traffic Classes")
    parser.add_argument("--src-data", type=Path, required=True, help="Thư mục nguồn chứa mask lớp hiếm")
    parser.add_argument("--dst-data", type=Path, required=True, help="Thư mục train đích để dán ảnh simulation")
    parser.add_argument("--classes", type=int, nargs="+", default=[3, 4, 6, 7], help="Danh sách ID lớp cần nhân bản")
    parser.add_argument("--max-paste", type=int, default=200, help="Số ảnh simulation tạo thêm")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_copy_paste(
        src_data=args.src_data,
        dst_data=args.dst_data,
        target_classes=set(args.classes),
        max_paste=args.max_paste,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
