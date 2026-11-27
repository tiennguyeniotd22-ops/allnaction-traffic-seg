# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Mọi bộ dữ liệu công khai (public dataset) đã dùng phải ghi rõ: **Tên, Đường dẫn, Giấy phép (License), Số ảnh đã dùng, và Cách ánh xạ về 11 lớp (`0..10`)**.

---

## Snapshot thực tế của lần 3 và pilot E1

| Nguồn | Train / val | Nguồn nhãn và giấy phép |
|---|---:|---|
| BDD100K | 180 / 45 | Ba nhóm ngày, đêm và mưa, mỗi nhóm 60 / 15. Box nguồn làm prompt SAM, sau đó nhóm sửa trong CVAT. Dữ liệu theo điều khoản Berkeley cho giáo dục, nghiên cứu, phi lợi nhuận. [Điều khoản](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst) |
| Vietnam Traffic Sign Detection v5 | 60 / 15 | Box biển báo và prompt detector cho các lớp khác, SAM tạo mask nháp, nhóm sửa trong CVAT. Bản xuất ghi Public Domain. [Trang phiên bản](https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8/dataset/5) |
| COCO val2017 | 320 / 80 | Ánh xạ car, bus, truck, bicycle, motorcycle, person, stop sign, traffic light. Người lái được ghép với xe theo quy tắc của bộ dựng. Annotation CC BY 4.0, ảnh giữ giấy phép Flickr riêng. [Điều khoản COCO](https://cocodataset.org/#termsofuse) |
| Ảnh tăng cường từ train COCO | 80 / 0 | 37 ảnh đêm và 43 ảnh mưa tổng hợp, giữ giấy phép ảnh cha. |

Tổng 640 / 140. SHA-256 bộ ZIP lần 3 bắt đầu `91268bb2`. E1 dùng cùng ảnh, điều chỉnh hình học theo annotation COCO, ZIP bắt đầu `d53ff10a`. Danh sách changed rows và checksum file cũ/mới nằm trong [sổ sửa E1](experiments/e1/edits.json).

**Nguồn đang chờ QC.** Poribohon-BD do Shaira Tabassum, Md. Sabbir Ullah, Nakib Hossain Al-nur và Swakkhar Shatabda công bố theo CC BY 4.0. Lấy 58 ảnh từ bản mẫu của tác giả, sàng lọc theo tên lớp, kích thước và dHash. Dùng 129 box CNG và 45 box rickshaw làm prompt SAM ViT-H cho lớp 3. Bảo toàn 174 mask PNG, 150 polygon đạt IoU với SAM ít nhất 0,93. Nhóm cần xem lại 24 mask khó chuyển thành polygon và duyệt đủ 11 lớp từng ảnh. Số ảnh nguồn này đã dùng trong E1 bằng 0. Manifest của gói lưu SHA gốc, SHA ảnh resize, đường dẫn XML, box, kích thước và tác giả. [Dataset và giấy phép](https://data.mendeley.com/datasets/pwyyg8zmk5/1) · [Bản mẫu tại commit nguồn](https://github.com/shairatabassum/BangladeshiVehiclesDataset/tree/d185287ebff01849089d560be0cc9f75305f89ee)

## Kế hoạch nguồn của các phiên bản cũ

| STT | Tên Bộ Dữ Liệu | Đường Dẫn (URL) | Giấy Phép (License) | Số Ảnh Đã Dùng (Train / Val) | Ghi Chú Xử Lý |
| :-: | :--- | :--- | :--- | :---: | :--- |
| **1** | **COCO 2017 (Traffic Subset)** | `https://cocodataset.org/`<br>`http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | **CC BY 4.0** (Annotations) | **v1:** `320 / 80`<br>**v2:** `800 / 200` | Lọc & ánh xạ tự động bằng `scripts/prepare_coco_traffic.py`; gộp `person + motorcycle/bicycle` thành `Motorcyclist`/`Cyclist`. |
| **2** | **Dữ liệu Tổng hợp Cảnh Khó (Đêm / Mưa)** | Sinh từ tập Train COCO qua `scripts/validate_and_augment.py` | Tự tạo từ tập Train (CC BY 4.0) | **v1:** `80 / 0`<br>**v2:** `200 / 0` | Chỉ đưa vào `train`, tuyệt đối không đưa vào `val`. |
| **3** | **Dashcam / Video Đường Phố VN (Tự thu thập)** | *(Điền link YouTube / nguồn tự quay khi bổ sung ở v3)* | Tự thu thập & gán nhãn | `0 / 0` *(dự kiến v3)* | Prelabel bằng `yolo26x-seg.pt` + sửa tay trên CVAT cho `Other vehicle`, `Sign`. |
| **4** | **IDD (Indian Driving Dataset)** *(nếu dùng thêm)* | `https://idd.insaan.iiit.ac.in/` | Non-Commercial / Research (IDD License) | `0 / 0` | Lấy thêm xe máy, `autorickshaw` (`Other vehicle`), `rider`. |
| **5** | **BDD100K (Instance Seg)** *(nếu dùng thêm)* | `https://bdd-data.berkeley.edu/` | BSD 3-Clause | `0 / 0` | Lấy thêm cảnh đêm, trời mưa. |
| **6** | **Roboflow Universe (VN Traffic)** *(nếu dùng thêm)* | `https://universe.roboflow.com/` *(điền link repo cụ thể)* | CC BY 4.0 / MIT | `0 / 0` | Bổ sung biển báo VN & xe ba gác/xích lô. |

---

## 2. Bảng Ánh Xạ Chi Tiết Về 11 Lớp Cuộc Thi (`0..10`)

| ID | Lớp Cuộc Thi | Ánh Xạ Từ **COCO 2017** (`prepare_coco_traffic.py`) | Ánh Xạ Từ **IDD** | Ánh Xạ Từ **BDD100K** |
| :-: | :--- | :--- | :--- | :--- |
| **0** | `Car` | `car` | `car` | `car` |
| **1** | `Bus` | `bus` | `bus` | `bus` |
| **2** | `Truck` | `truck` | `truck` | `truck` |
| **3** | `Other vehicle` | *(Chưa có — bổ sung từ Dashcam VN / IDD)* | `autorickshaw`, `caravan`, `trailer` | `train`, `trailer` |
| **4** | `Motorcycle` | `motorcycle` *(không có `person` ngồi trên)* | `motorcycle` *(không có `rider`)* | `motorcycle` *(không người)* |
| **5** | `Motorcyclist` | `person` + `motorcycle` *(gộp mask khi chồng lấn trên yên)* | `rider` + `motorcycle` *(gộp mask)* | `rider` + `motorcycle` *(gộp mask)* |
| **6** | `Bicycle` | `bicycle` *(không có `person` ngồi trên)* | `bicycle` *(không có `rider`)* | `bicycle` *(không người)* |
| **7** | `Cyclist` | `person` + `bicycle` *(gộp mask khi chồng lấn trên yên)* | `rider` + `bicycle` *(gộp mask)* | `rider` + `bicycle` *(gộp mask)* |
| **8** | `Pedestrian` | `person` *(không đi kèm xe 2 bánh)* | `person` | `pedestrian` |
| **9** | `Sign` | `stop sign` | `traffic sign` | `traffic sign` |
| **10** | `Traffic light` | `traffic light` | `traffic light` | `traffic light` |
