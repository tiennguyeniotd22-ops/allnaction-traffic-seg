"""
AllnAction RoboTaxi Hackathon — v2 Dataset Builder (COCO Traffic Subset -> 11 Classes YOLO-Seg)
v2 fixes:
- Eliminates self-intersecting polygons via rasterize-and-contour repair.
- Computes min_area_640 using aspect-preserving letterbox scale (640 / max(w, h)) matching notebook Step 2.
- Tightens rider-overlap rule to stop mislabeling Motorcycle (4) as Motorcyclist (5).
- Increases default subset to 1000 rich traffic images.

Usage:
    python scripts/prepare_coco_traffic.py --out dataset_traffic_v2 --max-images 1000 --zip
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

COCO_NAME_TO_COMP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "motorcycle": 4,
    "bicycle": 6,
    "person": 8,
    "stop sign": 9,
    "traffic light": 10,
}

PRIORITY_NAMES = {"motorcycle", "bicycle", "bus", "truck", "traffic light", "stop sign"}
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_annotations(cache_dir: Path) -> Path:
    """Download and extract COCO instances_val2017.json if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / "annotations" / "instances_val2017.json"
    if json_path.exists():
        return json_path

    zip_path = cache_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"[1/4] Đang tải COCO annotations (~253 MB) từ {ANNOTATIONS_URL}...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)

    print("[1/4] Đang giải nén instances_val2017.json...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("annotations/instances_val2017.json", cache_dir)
    return json_path


def repair_and_simplify_polygon(poly_norm: np.ndarray, grid: int = 640) -> np.ndarray:
    """Guarantee a simple, non-self-intersecting polygon via rasterize + outer contour."""
    if len(poly_norm) < 3:
        return poly_norm
    canvas = np.zeros((grid, grid), dtype=np.uint8)
    pts = np.round(np.clip(poly_norm, 0.0, 1.0) * (grid - 1)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.0015 * peri, True).reshape(-1, 2).astype(np.float32) / float(grid - 1)
    return np.clip(approx, 0.0, 1.0)


def polygon_area_at_640(poly_norm: np.ndarray, orig_w: int, orig_h: int) -> float:
    """Estimate mask pixel area after aspect-preserving resize to max_dim=640."""
    scale = 640.0 / float(max(orig_w, orig_h))
    pts = np.column_stack([poly_norm[:, 0] * orig_w * scale, poly_norm[:, 1] * orig_h * scale]).astype(np.float32)
    return float(cv2.contourArea(pts))


def box_overlap_rider(box_p: list[float], box_b: list[float]) -> bool:
    """Tighter rider check: person must overlap bike AND sit above/on the seat."""
    px1, py1, px2, py2 = box_p[0], box_p[1], box_p[0] + box_p[2], box_p[1] + box_p[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return False

    p_area = max(1e-6, box_p[2] * box_p[3])
    b_area = max(1e-6, box_b[2] * box_b[3])
    p_cx = 0.5 * (px1 + px2)
    p_cy = 0.5 * (py1 + py2)
    b_cy = 0.5 * (by1 + by2)
    return (
        (inter / min(p_area, b_area) >= 0.22)
        and (bx1 <= p_cx <= bx2)
        and (p_cy <= b_cy + 0.10 * box_b[3])
    )


def merge_polygons(poly_a: np.ndarray, poly_b: np.ndarray, grid: int = 640) -> np.ndarray:
    """Merge person + 2-wheeler normalized polygons into one clean outer contour."""
    canvas = np.zeros((grid, grid), dtype=np.uint8)
    for poly in (poly_a, poly_b):
        pts = np.round(np.clip(poly, 0.0, 1.0) * (grid - 1)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], 255)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return poly_b
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.0015 * peri, True).reshape(-1, 2).astype(np.float32) / float(grid - 1)
    return np.clip(approx, 0.0, 1.0)


def coco_seg_to_norm_poly(seg: list, w: int, h: int) -> np.ndarray | None:
    """Pick largest polygon part from COCO segmentation list and normalize to [0, 1]."""
    if not isinstance(seg, list) or not seg:
        return None
    largest = max(seg, key=len)
    if len(largest) < 6:
        return None
    pts = np.array(largest, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
    return pts


def download_one(url: str, dest: Path) -> bool:
    """Download single image if missing."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception:
        return False


def build_starter_dataset(
    out_dir: Path,
    cache_dir: Path,
    max_images: int,
    val_ratio: float,
    min_area_640: float,
    seed: int,
    make_zip: bool,
) -> None:
    """Build v2 YOLO-seg dataset from COCO traffic subset."""
    json_path = download_annotations(cache_dir)
    print(f"[2/4] Đang phân tích {json_path.name}...")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_info = {im["id"]: im for im in data["images"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 0 and cat_id_to_name.get(ann["category_id"]) in COCO_NAME_TO_COMP:
            anns_by_img[ann["image_id"]].append(ann)

    # Prioritize rare traffic classes (motorcycle, bicycle, stop sign, truck)
    scored_imgs = []
    for img_id, anns in anns_by_img.items():
        names = {cat_id_to_name[a["category_id"]] for a in anns}
        if not (names & (PRIORITY_NAMES | {"car"})):
            continue
        bonus = 15 * ("stop sign" in names) + 12 * ("motorcycle" in names) + 10 * ("bicycle" in names)
        score = bonus + len(names & PRIORITY_NAMES) * 10 + min(len(anns), 15)
        scored_imgs.append((score, img_id))

    scored_imgs.sort(key=lambda x: x[0], reverse=True)
    selected_ids = [img_id for _, img_id in scored_imgs[:max_images]]

    rng = np.random.default_rng(seed)
    rng.shuffle(selected_ids)
    n_val = max(1, int(len(selected_ids) * val_ratio))
    val_ids = set(selected_ids[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Đang tải {len(selected_ids)} ảnh giao thông (train={len(selected_ids)-n_val}, val={n_val})...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = []
        for img_id in selected_ids:
            info = img_id_to_info[img_id]
            split = "val" if img_id in val_ids else "train"
            dest = out_dir / "images" / split / info["file_name"]
            futs.append(pool.submit(download_one, info["coco_url"], dest))
        ok_count = sum(1 for fut in as_completed(futs) if fut.result())
    print(f"      Đã tải thành công {ok_count}/{len(selected_ids)} ảnh.")

    stats = {"train": Counter(), "val": Counter()}
    dropped_small = 0
    for img_id in selected_ids:
        info = img_id_to_info[img_id]
        split = "val" if img_id in val_ids else "train"
        img_path = out_dir / "images" / split / info["file_name"]
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
            stats[split][COMPETITION_CLASSES[cls_id]] += 1

        lbl_path = out_dir / "labels" / split / f"{Path(info['file_name']).stem}.txt"
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

    print(f"\n[4/4] === THỐNG KÊ BỘ DATA v2 (đã lọc {dropped_small} mask < {min_area_640}px) ===")
    for split in ("train", "val"):
        print(f"Split [{split.upper()}]: {dict(stats[split])}")

    if make_zip:
        zip_out = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"\nĐã đóng gói: {zip_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 Starter Traffic Dataset from COCO")
    parser.add_argument("--out", type=Path, default=Path("dataset_traffic_v2"))
    parser.add_argument("--cache", type=Path, default=Path("raw_cache"))
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-area", type=float, default=65.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    build_starter_dataset(
        out_dir=args.out,
        cache_dir=args.cache,
        max_images=args.max_images,
        val_ratio=args.val_ratio,
        min_area_640=args.min_area,
        seed=args.seed,
        make_zip=args.zip,
    )


if __name__ == "__main__":
    main()
