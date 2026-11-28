# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Ghi trung thực toàn bộ nguồn dữ liệu đã dùng (Tên, Đường dẫn, Giấy phép, Số ảnh thực dùng và Cách ánh xạ về 11 lớp `0..10`), tách bạch phần **đã đưa vào huấn luyện** và phần **đã chuẩn bị ngoài lề**.

## Nguồn công khai trong lineage của mô hình điểm cao nhất

`team21-mask-roi` đạt 27,54 mask mAP trên val140. Mô hình kế thừa ensemble 9 checkpoint từ 8 gói dữ liệu train (gói `e3-triage-v1` được dùng cho hai checkpoint). Danh sách checkpoint và SHA dữ liệu từng gói nằm trong [bảng kết quả](reports/TEAM21_KET_QUA.md) và [chỉ mục chạy lại](reports/replay/team21-mask-roi/REPLAY.md).

| Bộ dữ liệu công khai | Có trong lineage qua | Số lượng đã ghi nhận trong các phiên bản liên quan | Giấy phép / ánh xạ |
|---|---|---|---|
| BDD100K | `e3-triage-v1`, `e3-vnrepair-v1`, `team21-wideqc-v1` | 225 ảnh CVAT ban đầu; 43 crop semantic trong một bản mở rộng | Điều khoản Berkeley; ánh xạ từ nhãn giao thông sang 11 lớp ở manifest từng phiên bản. |
| Vietnam Traffic Sign Detection v5 | `e3-triage-v1` | 75 ảnh CVAT ban đầu | Trang bản xuất ghi Public Domain; biển ánh xạ sang `Sign (9)`, các lớp khác theo manifest. |
| COCO 2017 | `e3-triage-v1`, `team21-road-v1`, `team21-vistas-extra-v1` và ảnh cha của dữ liệu ghép | 31 ảnh rà toàn khung trong `team21-road-v1`; 69 crop trong `team21-vistas-extra-v1`; các phần COCO kế thừa được ghi theo manifest | Annotation CC BY 4.0; giấy phép ảnh Flickr được lưu theo từng ảnh. Gộp người đang ngồi với xe thành `Motorcyclist (5)`/`Cyclist (7)`; người dắt xe tách riêng. |
| Mapillary Vistas v2 | `e3-vnrepair-v1`, `team21-vistas-extra-v1`, `team21-paste-v1` | 212 crop trong bản Vistas trước; 182 cảnh bổ sung trong `team21-vistas-extra-v1`; ảnh nền của bản ghép giữ nguồn cha | CC BY-NC-SA 4.0; mask panoptic được ánh xạ theo tên lớp, sau đó kiểm polygon và quan hệ người–xe. |

Các số lượng trên mô tả từng phiên bản cụ thể, có phần kế thừa và chồng lặp nên không cộng thành tổng ảnh duy nhất. Manifest theo ảnh lưu nguồn, giấy phép, lớp đích và checksum; xem bảng phiên bản bên dưới cùng các file `*-manifest.json`. Các bộ đã khảo sát nhưng không có trong lineage này không được liệt kê là nguồn của mô hình điểm cao nhất.

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
