# Nhật Ký Thí Nghiệm — AllnAction RoboTaxi (YOLO26n-seg, 11 Lớp)

> **Quy tắc:** Ghi trung thực điểm chấm lại từ ONNX ZIP trên cùng cấu hình (`yolo26n-seg`, `640×640`, `opset=17`, FP32). File nặng (`best.pt`, `submission.zip`) lưu ngoài Git.

---

## 1. Bảng Tổng Hợp Tiến Triển Qua Các Phiên Bản

| Vòng / Nhánh | Bộ Dữ Liệu Thực Dùng | Tập Val | `Mask mAP@[.5:.95]` | `AP50` | `AP75` | Ghi Chú Chính |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **v1 Baseline** (`v1-baseline-21.92`) | `480` ảnh COCO (`400` train) | 80 COCO | **21,92** | 40,68 | 20,87 | `Other vehicle = 0`, `Motorcycle = 3,5` |
| **Round 3** (`round3-data-780-v1`) | `780` ảnh (`BDD 225` + `VN Sign 75` + `COCO 480`) | `val140` | **21,35** | 36,82 | 21,29 | Tập `val140` khắt khe hơn (`15` VN + `45` BDD + `80` Drive) |
| **Round 4** (`exp/round4-data`, 100 ep) | `780` ảnh (giữ nguyên split Round 3) | `val140` | **22,01** | 38,72 | — | Tăng `epochs` 20 → 100 (`SHA-256: 412fee1b...`) |
| **Pilot QC** (`team21-qc20`) | `1.156` ảnh đã sửa nhãn trên CVAT | `val140` | **24,13** | 41,55 | 24,49 | Rà 50 cảnh ưu tiên + 31 ảnh COCO kiểm trùng |
| **Đội 21 Tốt Nhất** (`team21-mean-wide-bn`) | `1.411 – 1.442` ảnh (`+212` Mapillary, `+43` BDD) | `val140` | **25,64** | **43,87** | **25,42** | Tăng `+1,51` so với Pilot (`[0,42; 2,64]`); `Other vehicle holdout = 61,61` |

---

## 2. Chi Tiết Cải Thiện Từng Lớp (`Pilot 24,13` → `team21-mean-wide-bn 25,64`)

| ID | Lớp | GT `val140` | AP Pilot | AP Bản Chọn (`25,64`) | Chênh Lệch |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 0 | `Car` | 546 | 40,27 | 41,96 | +1,69 |
| 1 | `Bus` | 35 | 39,52 | 42,17 | +2,65 |
| 2 | `Truck` | 82 | 19,91 | 23,85 | +3,94 |
| 3 | `Other vehicle` | 0 *(holdout: 7)* | *(50,98)* | *(61,61)* | *+10,63 (holdout)* |
| 4 | `Motorcycle` | 54 | 7,45 | 9,52 | +2,07 |
| 5 | `Motorcyclist` | 70 | 35,07 | 34,15 | -0,92 |
| 6 | `Bicycle` | 44 | 13,45 | 13,76 | +0,31 |
| 7 | `Cyclist` | 33 | 15,66 | 14,92 | -0,74 |
| 8 | `Pedestrian` | 418 | 27,10 | 29,04 | +1,94 |
| 9 | `Sign` | 152 | 27,26 | 28,74 | +1,48 |
| 10 | `Traffic light` | 154 | 15,58 | 18,28 | +2,70 |
