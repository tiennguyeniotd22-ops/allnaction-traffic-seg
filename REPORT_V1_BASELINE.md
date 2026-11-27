# Báo Cáo Đánh Giá Phiên Bản 1 (Version 1 — Baseline)

- **Git Tag:** `v1-baseline-21.92` (commit `7dfc18f`)
- **Mã đội / Seed:** `TEAM_ID = 26` (`SEED = 26`)
- **Thiết bị huấn luyện:** Google Colab — `GPU Tesla T4`
- **Điểm đạt được trên `val`:** **`mask mAP@[.5:.95] = 0.2192`** (**`21.92` điểm**)

---

## 1. Cấu Trúc Mô Hình & Tham Số Huấn Luyện (v1)

| Thành Phần | Thông Số Thực Tế Quan Sát Được |
| :--- | :--- |
| **Kiến trúc** | `YOLO26n-seg` (`Ultralytics 8.4.214`) — `137` layers, `2,706,213` parameters, `9.2 GFLOPs` |
| **Định dạng xuất (ONNX)** | `best.onnx` (`10.6 MB`, `opset=17`, `simplify=True`, `fp32`), đóng gói `submission.zip` (`9.2 MB` < `42 MiB`) |
| **Kích thước Input / Output** | Input `[1, 3, 640, 640]` → Output0 `[1, 47, 8400]`, Output1 `[1, 32, 160, 160]` *(mask prototype `160×160`)* |
| **Tốc độ suy luận CPU (2 luồng)** | `p50 = 246 ms/ảnh` (model) + `146 ms/ảnh` (giải mã mask) |
| **Tham số Train v1** | `epochs=60`, `batch=16`, `imgsz=640`, `lr0=0.01`, `cos_lr=False`, `copy_paste=0.0`, `close_mosaic=10`, `erasing=0.4` |

---

## 2. Kết Quả Fit Trên Tập `val` (80 Ảnh / 951 Đối Tượng)

- **Tổng hợp:** `mask mAP@[.5:.95] = 0.2192` (`21.92` điểm) | `AP50 = 0.4068` | `AP75 = 0.2087` | `23` nhãn nhỏ bị bỏ qua.

| ID | Lớp | Số Nhãn Train | Số Nhãn Val | `AP[.5:.95]` | `AP50` | Trạng Thái |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | `Car` | 1.060 | 211 | 0.303 | 0.560 | Khá |
| 1 | `Bus` | 223 | 25 | **0.469** | 0.604 | **Tốt nhất** |
| 2 | `Truck` | 251 | 61 | 0.184 | 0.290 | Trung bình |
| 3 | `Other vehicle` | **0** | **0** | — | — | **Thiếu hoàn toàn** |
| 4 | `Motorcycle` | 136 | 40 | **0.035** | **0.119** | **Cực yếu** |
| 5 | `Motorcyclist` | 128 | 29 | 0.273 | 0.559 | Khá |
| 6 | `Bicycle` | 135 | 40 | **0.129** | 0.305 | **Yếu** |
| 7 | `Cyclist` | 76 | 23 | 0.173 | 0.303 | Trung bình (ít mẫu) |
| 8 | `Pedestrian` | 2.066 | 388 | 0.291 | **0.607** | Khá |
| 9 | `Sign` | 29 | 7 | **0.160** | 0.270 | **Yếu (rất ít mẫu)** |
| 10 | `Traffic light` | 422 | 104 | 0.176 | 0.453 | Trung bình |

---

## 3. Đánh Giá: Điểm Tốt vs. Điểm Kém Ở Version 1

### Điểm Đã Làm Tốt
1. **Thông toàn bộ luồng End-to-End hợp lệ:** Từ nạp data → kiểm tra → train T4 GPU → xuất `best.onnx` (`10.6 MB`) → vượt qua toàn bộ kiểm tra hệ thống chấm (`submission.zip` `9.2 MB`).
2. **Nhận diện tốt các lớp kích thước lớn & đủ mẫu:** `Bus` (`0.469`), `Car` (`0.303`), `Pedestrian` (`0.291`), `Motorcyclist` (`0.273`).
3. **Rule gộp `Motorcyclist` bước đầu có hiệu quả:** Dù COCO không có lớp `Motorcyclist`, việc gộp `person + motorcycle` giúp lớp này đạt `AP50 = 0.559`.

### Điểm Kém & Nguyên Nhân Gốc Rễ (Căn Cứ Cho Version 2)
1. **Lỗi chất lượng nhãn (Cell 13):**
   - Tồn tại **5 đa giác tự cắt nhau (bị xoắn)** ở tập `train` và **156 nhãn `< 64 px`** (`Pedestrian=70`, `Car=44`, `Traffic light=34`, `Bicycle=6`, `Bus=1`, `Sign=1`).
   - *Nguyên nhân:* `prepare_coco_traffic.py` v1 tính diện tích theo `640×640` méo tỷ lệ thay vì giữ tỷ lệ cạnh (`640 / max(w, h)`).
2. **`Motorcycle (4)` cực yếu (`AP[.5:.95] = 0.035`, `AP50 = 0.119`):**
   - *Nguyên nhân:* Rule gộp v1 quá lỏng (`inter >= 0.15`), gộp nhầm người đứng gần xe thành `Motorcyclist` hoặc gây nhiễu phân biệt giữa xe máy có người và không người.
3. **Độ chính xác biên mask kém ở ngưỡng IoU cao (`AP50 = 0.4068` nhưng `AP75 = 0.2087` — rớt 48.7%):**
   - *Nguyên nhân:* `EPOCHS=60` hơi ngắn, `copy_paste=0.0` (chưa tăng cường lớp hiếm), `mask_ratio=4` mặc định (prototype `160×160` hơi thô cho vật nhỏ), chưa bật `AdamW` + `cos_lr`.
4. **Thiếu lớp & lệch phân phối Việt Nam:**
   - `3: Other vehicle = 0` mẫu (sẽ mất `1/11` điểm trên tập test thật của BTC).
   - `9: Sign = 29` mẫu (dưới ngưỡng `100`), thiếu biển báo và xe ba gác/xích lô Việt Nam.

---

## 4. Kế Hoạch Hành Động Cho Version 2 (Từ Các Căn Cứ Trên)
1. **Sửa sạch dữ liệu:** Dùng `repair_and_simplify_polygon()` (rasterize + `findContours`) để đạt `0` đa giác tự cắt; lọc sạch nhãn `< 64 px` theo đúng tỷ lệ `640 / max(w, h)`.
2. **Siết chặt rule tách `Motorcycle` vs `Motorcyclist`:** Yêu cầu `inter >= 0.22` và tâm người nằm trên yên xe (`p_cy <= b_cy + 0.10 * bh`).
3. **Mở rộng quy mô & bổ sung cảnh Việt Nam:** Tăng tập COCO lên `1.000` ảnh + `25%` ảnh đêm/mưa (`v2`), sau đó bổ sung ảnh Dashcam VN cho `Other vehicle` (`>= 100` mẫu) và `Sign` (`>= 100` mẫu) ở `v3`.
4. **Tinh chỉnh tham số Train (`TRAIN_ARGS`):** `epochs=100`, `optimizer="AdamW"`, `lr0=0.002`, `cos_lr=True`, `copy_paste=0.35`, `close_mosaic=15`, `mask_ratio=2`.
