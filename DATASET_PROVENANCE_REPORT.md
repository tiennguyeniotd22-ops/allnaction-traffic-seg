# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Mọi bộ dữ liệu công khai (public dataset) đã dùng phải ghi rõ: **Tên, Đường dẫn, Giấy phép (License), Số ảnh đã dùng, và Cách ánh xạ về 11 lớp (`0..10`)**.

---

## Dữ liệu bổ sung sau vòng 4

Tất cả số lượng dưới đây tính theo ảnh hoặc crop đưa vào train. Validation 140 giữ nguyên ảnh và nhãn. Các bản tăng cường giữ giấy phép của ảnh cha.

| Phiên bản | Thành phần bổ sung | Nguồn nhãn và ánh xạ |
|---|---|---|
| E2, 802 train | 126 COCO train2017, 32 Poribohon-BD, 4 crop biển Việt Nam v5 | COCO dùng mask instance gốc. Với Poribohon, box nguồn làm prompt SAM, duyệt lớp 3 theo loại xe và kiểm các đối tượng khác trong crop. 6 ảnh/crop Poribohon khác được khóa riêng cho holdout. |
| E3 Việt Nam, 834 train | 7 crop camera giao thông từ Vietnam Traffic Detection v2 và 25 crop từ Traffic Sign in Ho Chi Minh City v4 | Box thành prompt, xem lại đường biên và lớp. Thêm 10 Car, 22 Motorcyclist, 26 Sign. Camera ban ngày có trường hợp tên lớp trong nguồn đảo ô tô và xe máy. Metadata của hai phiên bản khai báo Apache 2.0. [Vietnam Traffic Detection](https://www.kaggle.com/datasets/huanhuyle/vietnam-traffic-detection) · [Traffic Sign in Ho Chi Minh City](https://www.kaggle.com/datasets/nhduynguyn/traffic-sign-in-ho-chi-minh-city) |
| E3 sửa taxonomy, 1.113 train | Giữ thêm 242 COCO train2017 và 37 cửa sổ KITTI instance segmentation 2015 sau duyệt | COCO lưu license Flickr riêng cho từng ảnh, annotation CC BY 4.0. KITTI ghi CC BY-NC-SA 3.0. Dùng mask gốc, ghép rider đã kiểm với xe, tách cụm biển và đèn, đối chiếu ảnh cha với split và holdout. Sổ sửa ghi checksum từng label thay đổi. [COCO](https://cocodataset.org/#termsofuse) · [KITTI](https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015) · [Sổ sửa](experiments/overnight/taxonomy-decisions.json) |

## Dữ liệu vòng 4 ngày 28/11/2026

Vòng 4 dùng lại trọn bộ 780 ảnh của vòng 3, gồm 225 ảnh BDD100K, 75 ảnh Vietnam Traffic Sign Detection v5 và 480 ảnh COCO hoặc biến thể. ZIP dữ liệu có SHA-256 `91268bb2a4b520de7cb9eb1f2f9d59b740671a539ab73d1d3bba300e9d51e7a6`. 640 ảnh dùng để train, 140 ảnh để so checkpoint. Các bộ Poribohon-BD, LVIS và các ảnh biển báo bổ sung vẫn nằm ngoài lần train này. [Số ảnh, instance và bộ tham số vòng 4](experiments/round4/run_report.json)

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
