# Nhật Ký Thí Nghiệm — AllnAction RoboTaxi (YOLO26n-seg, 11 Lớp)

> **Quy tắc:** Mỗi lần chạy xong `train_and_export.ipynb`, cập nhật 1 dòng vào bảng dưới đây và đẩy lên GitHub. File nặng (`best.pt`, `submission.zip`) lưu theo thư mục version trên Google Drive (`DRIVE_OUTPUT_DIR`).

---

## 1. Bảng Theo Dõi Điểm Số Qua Các Vòng

| Vòng | Git Tag / Commit | Bộ Dữ Liệu | Thay Đổi Chính (Data & Tham Số) | `mAP@[.5:.95]` (×100) | `AP50` | `AP75` | Lớp Yếu Nhất & Ghi Chú |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **v1** | `v1-baseline-21.92` (`7dfc18f`) | `dataset_traffic_11.zip` (400 train, 80 val COCO + aug đêm/mưa) | Baseline đầu tiên (`EPOCHS=60`, `BATCH=16`, `copy_paste=0.0`) | **21.92** | 0.4068 | 0.2087 | `Other vehicle` (0 mẫu), `Motorcycle` (0.035), `Bicycle` (0.129), `Sign` (0.160). Có 5 đa giác tự cắt & 156 nhãn `<64px`. |
| **v2** | `___` | `dataset_traffic_v2.zip` | *(Dự kiến)* Lọc sạch đa giác lỗi, tăng data + thêm ảnh VN (`Other vehicle`, `Sign`), `copy_paste=0.35`, `mask_ratio=2`, `AdamW`, `EPOCHS=100` | `___` | `___` | `___` | `___` |
| **v3** | `___` | `dataset_traffic_v3.zip` | *(Dự kiến)* Active Learning sửa 6 ảnh lỗi nhất từ v2 | `___` | `___` | `___` | `___` |

---

## 2. Chi Tiết Điểm Từng Lớp (Lượt 1 — `21.92`)

| ID | Lớp | `AP[.5:.95]` | `AP50` | Số Nhãn Val | Đánh Giá |
| :-: | :--- | :---: | :---: | :---: | :--- |
| 0 | `Car` | 0.303 | 0.560 | 211 | Khá |
| 1 | `Bus` | 0.469 | 0.604 | 25 | Tốt nhất |
| 2 | `Truck` | 0.184 | 0.290 | 61 | Trung bình |
| 3 | `Other vehicle` | — | — | 0 | **Thiếu hoàn toàn (cần bổ sung gấp)** |
| 4 | `Motorcycle` | **0.035** | 0.119 | 40 | **Cực yếu (nhiễu với Motorcyclist)** |
| 5 | `Motorcyclist` | 0.273 | 0.559 | 29 | Khá |
| 6 | `Bicycle` | **0.129** | 0.305 | 40 | **Yếu** |
| 7 | `Cyclist` | 0.173 | 0.303 | 23 | Trung bình |
| 8 | `Pedestrian` | 0.291 | 0.607 | 388 | Khá |
| 9 | `Sign` | **0.160** | 0.270 | 7 | **Yếu (ít mẫu)** |
| 10 | `Traffic light` | 0.176 | 0.453 | 104 | Trung bình |

---

## 3. Nơi Lưu Trữ Artifact Nặng (Google Drive)
- `AllnAction_Runs/v1_baseline/`: chứa `best.pt`, `submission.zip` (9.2 MB), `run_report.json`.
- `AllnAction_Runs/v2/`: chứa kết quả Lượt 2 (điền vào `DRIVE_OUTPUT_DIR` ở Cell 6 trước khi chạy).
