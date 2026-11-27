# Nhật Ký Thí Nghiệm — AllnAction RoboTaxi (YOLO26n-seg, 11 Lớp)

## So sánh qua đêm trên validation cố định

Các số dưới đây là điểm mask trên thang 100. 140 ảnh gồm 60 ảnh nhóm và 80 ảnh Drive. 15 ảnh Việt Nam nằm trong 60 ảnh nhóm. Mỗi lượt dùng seed 26, cùng weights khởi tạo và cùng notebook.

| Lượt | Train | Epoch | mAP 140 | AP50 | AP75 | Việt Nam 15 | Nhóm 60 | Drive 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vòng 4 | 640 | 100 | 22,01 | 38,72 | 22,50 | 18,79 | 23,96 | 18,19 |
| E1 geometry100 | 640 | 100 | 22,10 | 39,24 | 21,38 | 15,96 | 22,35 | 19,04 |
| E2 pilot | 802 | 20 | 21,14 | 37,89 | 20,61 | 15,29 | 19,59 | 19,64 |

E1 vượt vòng 4 0,09 điểm, với khoảng bootstrap ghép cặp 95% từ −1,27 đến +1,23. E2 pilot vượt E1 cùng 20 epoch 0,35 điểm, với khoảng từ −1,08 đến +2,12. Val140 có 0 GT lớp 3. Trên holdout 6 ảnh, 7 GT, 3 nhóm cảnh, E2 pilot đạt AP lớp 3 41,00 và AP50 49,27. Hai checkpoint 100 epoch trước E2 có AP lớp 3 bằng 0. Toàn bộ AP theo lớp, điểm từng nguồn, checksum ZIP, tài nguyên GPU và kết quả đối chiếu ONNX có trong [bảng điểm máy đọc](experiments/overnight/verified-scores.json).

**Bộ train sửa taxonomy.** Rà 493 polygon Truck và 58 cặp polygon chồng lấn trên train. Đã đổi 116 nhãn thành Car, 4 thành Bus, 8 thành Other vehicle, bỏ 52 annotation trùng và loại 4 ảnh mới có mask nhập hai vật thể hoặc che khuất sai. Sửa 26 polygon train tự cắt, IoU raster sau lưu đạt 1,00 cho cả 26. Còn 11 polygon tự cắt trong GT validation đã khóa. Bộ mới có 1.113 ảnh train, 10.036 instance train và 140 cặp validation khớp SHA. So sánh với E3 Việt Nam sẽ đo tác động gộp của bổ sung ảnh và sửa nhãn. [Quyết định theo dòng nguồn](experiments/overnight/taxonomy-decisions.json) · [Manifest kiểm chứng](experiments/overnight/taxonomy-manifest.json)

## Vòng 4 hoàn tất ngày 28/11/2026

Train đủ 100 epoch trên L4 với 640 ảnh train. Hình ảnh và nhãn của 140 ảnh validation khớp checksum vòng 3. Thay đổi epoch từ 20 lên 100 và patience từ 20 lên 25, giữ seed 26, batch 8. Trên bộ chấm ONNX của notebook, mask mAP@[.50:.95] tăng từ 21,35 lên **22,01**, AP50 từ 36,82 lên **38,72**. Khoảng bootstrap 95% của chênh lệch mAP theo cảnh là −0,45 đến +2,20 điểm. Riêng 15 ảnh Việt Nam đạt 18,79 so với 18,85. Lớp Other vehicle có 0 GT. Điểm public của ZIP vòng 4 đang chờ đội nộp.

File ZIP có 9.660.994 byte, một entry `model.onnx` ở gốc, SHA-256 `412fee1ba894c34b2946703888821df8b29b2ff48b6ec4e8f6ad325ae1dc2c47`. ONNX opset 17 chạy được trên CPU và đã đối chiếu hai đầu ra với PyTorch trên ba ảnh. [Báo cáo notebook](experiments/round4/run_report.json) · [So sánh trên cùng 140 ảnh](experiments/round4/comparison.json)

## Kết quả kiểm chứng tối 27/11/2026

**Giữ checkpoint lần 3.** Điểm dưới đây dùng ONNX và evaluator nhúng của BTC trên cùng 140 ảnh. Đơn vị là điểm trên thang 100. Tất cả ảnh và GT validation đã đối chiếu bằng SHA-256. Pilot E1 thay 36 đường biên ở 32 ảnh train, gồm 35 instance gốc và một bản tăng cường có nhãn trùng khớp. Giữ nguyên số instance, tên lớp, ảnh, split, pretrained weights và toàn bộ tham số huấn luyện. 35 ca được chọn sau khi Codex kiểm tra cặp overlay trước/sau, chưa qua duyệt chéo của thành viên.

| Chỉ số | Lần 3 | Pilot E1 |
|---|---:|---:|
| Mask mAP@[.50:.95], 140 ảnh | 21,35 | 20,79 |
| AP50 | 36,82 | 36,08 |
| AP75 | 21,29 | 20,53 |
| Mask mAP, 60 ảnh nhóm | 23,45 | 21,41 |
| Mask mAP, 80 ảnh Drive | 18,25 | 18,51 |
| Mask mAP, riêng 15 ảnh Việt Nam | 18,85 | 16,41 |

Chênh lệch toàn tập là −0,56 điểm. Khoảng bootstrap ghép cặp theo cảnh và điều kiện, 300 lần lấy mẫu lại, là [−1,80; +0,70]. Một seed và 15 ảnh Việt Nam chỉ đủ làm căn cứ giữ mốc cũ. Thiếu GT lớp 3 khiến cả hai điểm trung bình local chỉ bao phủ 10 lớp. Điểm public 45,70 trên giao diện và mAP chi tiết 10,81 chưa truy được về checksum của một bài nộp.

| Lớp | Instance train / val | AP lần 3 | AP E1 |
|---|---:|---:|---:|
| Car | 2696 / 623 | 39,43 | 38,37 |
| Bus | 252 / 35 | 32,19 | 36,23 |
| Truck | 350 / 83 | 21,82 | 19,55 |
| Other vehicle | 0 / 0 | thiếu GT | thiếu GT |
| Motorcycle | 152 / 54 | 6,28 | 6,22 |
| Motorcyclist | 170 / 70 | 33,14 | 27,74 |
| Bicycle | 142 / 45 | 4,75 | 4,26 |
| Cyclist | 99 / 33 | 15,01 | 13,77 |
| Pedestrian | 2286 / 432 | 24,31 | 24,37 |
| Sign | 741 / 218 | 22,29 | 22,99 |
| Traffic light | 890 / 228 | 14,28 | 14,40 |

**Tái lập E1.** Commit huấn luyện `7ac0061`. Seed 26, YOLO26n-seg 640, batch 8, 20 epoch, AdamW lr0 0,002, cosine, mask ratio 2, Copy-Paste 0,35, close mosaic 15. L4 mất 461 giây cho fit và native validation, PyTorch ghi đỉnh cấp phát 4.949 MiB và giữ chỗ 10.124 MiB. ZIP có một tệp model.onnx, 9.665.818 byte, opset 17. ONNX checker và đối chiếu hai tensor đầu ra với PyTorch trên ba ảnh đã qua. Hàm diện tích mới khớp chính xác với evaluator cho 9.599 mask, gồm 969 mask train và 233 mask val dưới 64 px².

Tại thư mục VM `~/baseline-cvat300`, với các script trong checkout và pretrained weights đã ghi hash, dùng thư mục đầu ra mới cho mỗi lần chạy.

```bash
.venv/bin/python e1_geometry/run_notebook_pilot.py --dataset e1_geometry/pilot.zip --run e1_geometry/replay20 --revision 7ac00613a4bae7ec4e83e2d785f580d0fb9e6429
.venv/bin/python e1_geometry/compare_pilot.py --baseline team_round3/train20 --pilot e1_geometry/replay20 --notebook train_and_export.ipynb --manifest audit_phase12/validation-manifest.json --out e1_geometry/replay-comparison.json
```

**Vòng tiếp theo.** E2a bổ sung các ảnh lớp 3 đã được duyệt toàn khung vào bộ lần 3. E2b thử riêng biển báo Việt Nam. Cần một tập đo lớp 3 độc lập đã khóa và chấm lại E0 trên đó. Sau đó mới thử ca hai bánh, sampling, Copy-Paste, mask ratio và optimizer trong các run riêng. Cấu hình full training đề xuất sau khi có pilot đạt là tối đa 100 epoch, patience 25, batch 8, seed 26, còn lại theo ablation thắng. Cấu hình đang chờ trưởng nhóm duyệt.

[Báo cáo notebook](experiments/e1/run_report.json) · [So sánh chính thức và kiểm tra ONNX](experiments/e1/comparison.json) · [Tài nguyên và checksum](experiments/e1/resources.json) · [Danh sách sửa từng mask](experiments/e1/edits.json)

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
