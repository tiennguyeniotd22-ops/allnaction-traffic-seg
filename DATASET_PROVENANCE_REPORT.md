# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Mọi bộ dữ liệu công khai (public dataset) đã dùng phải ghi rõ: **Tên, Đường dẫn, Giấy phép (License), Số ảnh đã dùng, và Cách ánh xạ về 11 lớp (`0..10`)**.

---

## 1. Bảng Tổng Hợp Các Nguồn Dữ Liệu Đã Dùng

| STT | Tên Bộ Dữ Liệu | Đường Dẫn (URL) | Giấy Phép (License) | Số Ảnh Đã Dùng (Train / Val) | Ghi Chú Xử Lý |
| :-: | :--- | :--- | :--- | :---: | :--- |
| 1 | **IDD (Indian Driving Dataset)** | `https://idd.insaan.iiit.ac.in/` | Non-Commercial / Research (IDD License) | `___ / ___` | Lấy ảnh đông xe máy, xe ba gác (`autorickshaw`), `rider`. |
| 2 | **BDD100K (Instance Seg)** | `https://bdd-data.berkeley.edu/` | BSD 3-Clause | `___ / ___` | Lấy thêm cảnh đêm, trời mưa, đường phố đông. |
| 3 | **Mapillary Vistas** | `https://www.mapillary.com/dataset/vistas` | CC BY-NC-SA 4.0 | `___ / ___` | Dùng cho `Motorcyclist`, `Cyclist`, `Sign`, `Traffic light`. |
| 4 | **Roboflow Universe (VN Traffic)** | `https://universe.roboflow.com/...` *(điền link cụ thể)* | CC BY 4.0 / MIT *(kiểm tra từng repo)* | `___ / ___` | Bổ sung biển báo VN & xe đặc thù. |
| 5 | **Dashcam / Video Đường Phố VN (Tự thu thập)** | Nguồn tự quay / YouTube *(ghi link nếu có)* | Tự thu thập & gán nhãn | `___ / ___` | Prelabel bằng `yolo26x-seg.pt` + người sửa lại trên CVAT. |
| 6 | **Dữ liệu Tổng hợp & Aug (Mưa / Đêm / Copy-Paste)** | Sinh từ `scripts/validate_and_augment.py` | Tự tạo từ tập Train | `___ / 0` | Chỉ đưa vào `train`, tuyệt đối không đưa vào `val`. |

---

## 2. Bảng Ánh Xạ Chi Tiết Về 11 Lớp Cuộc Thi (`0..10`)

| ID | Lớp Cuộc Thi | Ánh Xạ Từ **IDD** | Ánh Xạ Từ **BDD100K** | Ánh Xạ Từ **Mapillary Vistas** | Ánh Xạ Từ **COCO Prelabel** |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **0** | `Car` | `car` | `car` | `Car` | `car` |
| **1** | `Bus` | `bus` | `bus` | `Bus` | `bus` |
| **2** | `Truck` | `truck` | `truck` | `Truck` | `truck` |
| **3** | `Other vehicle` | `autorickshaw`, `caravan`, `trailer`, `vehicle fallback` | `train`, `trailer` *(hoặc gán tay xe ba gác/xích lô)* | `Other Vehicle`, `Wheeled Slow` | *(Gán/sửa tay trên CVAT)* |
| **4** | `Motorcycle` | `motorcycle` *(không có `rider` chồng lên)* | `motorcycle` *(không người)* | `Motorcycle` *(không người)* | `motorcycle` *(không giao `person`)* |
| **5** | `Motorcyclist` | `rider` + `motorcycle` *(gộp mask)* | `rider` + `motorcycle` *(gộp mask)* | `Motorcyclist` | `person` + `motorcycle` *(gộp mask)* |
| **6** | `Bicycle` | `bicycle` *(không người)* | `bicycle` *(không người)* | `Bicycle` *(không người)* | `bicycle` *(không giao `person`)* |
| **7** | `Cyclist` | `rider` + `bicycle` *(gộp mask)* | `rider` + `bicycle` *(gộp mask)* | `Bicyclist` | `person` + `bicycle` *(gộp mask)* |
| **8** | `Pedestrian` | `person` | `pedestrian` | `Person` | `person` *(không đi kèm xe 2 bánh)* |
| **9** | `Sign` | `traffic sign` | `traffic sign` | `Traffic Sign (Front/Back)` | `stop sign` + *(SAM gán thêm)* |
| **10** | `Traffic light` | `traffic light` | `traffic light` | `Traffic Light` | `traffic light` |
