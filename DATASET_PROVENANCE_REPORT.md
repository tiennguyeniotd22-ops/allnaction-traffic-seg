# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Ghi trung thực toàn bộ nguồn dữ liệu đã dùng (Tên, Đường dẫn, Giấy phép, Số ảnh thực dùng và Cách ánh xạ về 11 lớp `0..10`), tách bạch phần **đã đưa vào huấn luyện** và phần **đã chuẩn bị ngoài lề**.

---

## 1. Dữ Liệu ĐÃ THỰC SỰ DÙNG Trong Các Bản Huấn Luyện (Đã Xác Minh SHA-256)

### A. Bộ Lõi Round 3 & Round 4 (`780 ảnh = 640 train / 140 val`, SHA-256: `91268bb2...`)
| Nguồn Dữ Liệu | Đường Dẫn (URL) | Giấy Phép | Train / Val | Cách Xử Lý & Ánh Xạ |
| :--- | :--- | :--- | :---: | :--- |
| **BDD100K** | `https://huggingface.co/datasets/dronefreak/BDD100K`<br>`https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst` | BSD 3-Clause (Giáo dục/NC) | `180 / 45` | Chia đều ngày/đêm/mưa (`60/15` mỗi nhóm); box gốc → SAM ViT-H → sửa trên CVAT. |
| **Vietnam Traffic Sign Detection v5** | `https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8/dataset/5` | Public Domain | `60 / 15` | YOLOv7 + box biển báo → SAM ViT-H → sửa trên CVAT (`9: Sign` & các lớp xe VN). |
| **COCO 2017 (`val2017`)** | `https://cocodataset.org/#termsofuse` | CC BY 4.0 (Ann.) | `320 / 80` | Lọc 8 lớp giao thông (`prepare_coco_traffic.py`), tự gộp `person` + xe 2 bánh thành `Motorcyclist`/`Cyclist`. |
| **Ảnh tăng cường Đêm / Mưa (COCO)** | Sinh từ tập Train COCO qua `scripts/validate_and_augment.py` | CC BY 4.0 | `80 / 0` | `37` ảnh đêm + `43` ảnh mưa tổng hợp, chỉ đưa vào `train`. |

### B. Bộ Mở Rộng Đội 21 (`exp/overnight-qc` → `team21-mean-wide-bn`: `25,64`)
- Giữ nguyên **780 ảnh gốc** (`640 / 140`) + **`31` ảnh COCO** đã duyệt (`1.156` ảnh) → mở rộng lên **`1.411 – 1.442` ảnh train** gồm **`212` crop Mapillary Vistas** (`CC BY-NC-SA 4.0`, `https://www.mapillary.com/dataset/vistas`), **`43` crop BDD100K**, **`24` tệp nhãn VN đã sửa** và **`48` tệp QC chung**.
- Tập `val140` giữ cố định (`45` BDD + `15` VN + `80` Drive); đánh giá riêng lớp `3: Other vehicle` trên holdout `6` ảnh / `7` GT.

---

## 2. Dữ Liệu ĐÃ CHUẨN BỊ / PRELABEL SẴN (Chưa Đưa Vào Bản Train Chính Thức `0` Ảnh)

| Nguồn Dữ Liệu | Đường Dẫn (URL) | Giấy Phép | Số Ảnh Đã Tải | Trạng Thái |
| :--- | :--- | :--- | :---: | :--- |
| **Poribohon-BD** | `https://data.mendeley.com/datasets/pwyyg8zmk5/1` | CC BY 4.0 | `58` cảnh (`174` mask) | Đã chạy SAM ViT-H cho `Other vehicle`; chờ QC thêm vào vòng sau (`0` ảnh trong bản nộp). |
| **COCO 2017 (`train2017` subset)** | `https://cocodataset.org/` | CC BY 4.0 | `400` + `80` đêm/mưa | Đã đóng gói `dataset_supplement_v4.zip` (`Motorcycle`, `Bicycle`, `Cyclist`); chưa gộp bản nộp. |
| **Wikimedia Commons (VN & Other vehicle)** | `https://commons.wikimedia.org/` *(chi tiết trong `manifest_*.csv`)* | CC BY / Public Domain | `50` Other vehicle + `24` đèn VN + `4` synth | Đã tải & chuẩn bị notebook SAM 3 / SAM 2.1; chưa đưa vào bản train cuối. |

---

## 3. Bảng Ánh Xạ Chi Tiết Về 11 Lớp Cuộc Thi (`0..10`)

| ID | Lớp Cuộc Thi | Ánh Xạ Từ **COCO 2017** | Ánh Xạ Từ **BDD100K / VN Sign** | Ánh Xạ Từ **Mapillary Vistas / Poribohon** |
| :-: | :--- | :--- | :--- | :--- |
| **0** | `Car` | `car` | `car` | `Car` |
| **1** | `Bus` | `bus` | `bus` | `Bus` |
| **2** | `Truck` | `truck` | `truck` | `Truck` |
| **3** | `Other vehicle` | *(Không có)* | *(Gán bổ sung xe ba gác/xích lô/rơ-moóc)* | `Other Vehicle` / `CNG`, `rickshaw` |
| **4** | `Motorcycle` | `motorcycle` *(trống)* | `motorcycle` *(trống / đang dắt)* | `Motorcycle` |
| **5** | `Motorcyclist` | `person` + `motorcycle` *(gộp)* | `rider` + `motorcycle` *(gộp)* | `Motorcyclist` |
| **6** | `Bicycle` | `bicycle` *(trống)* | `bicycle` *(trống / đang dắt)* | `Bicycle` |
| **7** | `Cyclist` | `person` + `bicycle` *(gộp)* | `rider` + `bicycle` *(gộp)* | `Bicyclist` |
| **8** | `Pedestrian` | `person` *(không trên xe)* | `pedestrian` | `Person` |
| **9** | `Sign` | `stop sign` | `traffic sign` / VN Sign | `Traffic Sign` |
| **10** | `Traffic light` | `traffic light` | `traffic light` | `Traffic Light` |
