# Báo Cáo Nguồn Dữ Liệu Huấn Luyện (Dataset Provenance & License Log)

> **Quy định cuộc thi:** Mọi bộ dữ liệu công khai (public dataset) đã dùng phải ghi rõ: **Tên, Đường dẫn, Giấy phép (License), Số ảnh đã dùng, và Cách ánh xạ về 11 lớp (`0..10`)**.

---

## Dữ liệu bổ sung sau vòng 4

Bộ `team21-paste-v1` bổ sung 101 ảnh ghép từ các ảnh cha thuộc tập train, nâng số ảnh train lên 1.794. Nhóm giữ 93 mẫu xe hai bánh sau khi xem 236 mẫu cắt. Trong 164 cảnh ghép đã kiểm tra, nhóm nhận 13 cảnh xe máy không người, 32 cảnh xe máy có người, 20 cảnh xe đạp không người và 36 cảnh xe đạp có người. Mỗi cảnh giữ polygon của ảnh nền và thêm một polygon của đối tượng ghép. Nền lấy từ Mapillary Vistas v2, mẫu cắt lấy từ các phiên bản train ghi trong sổ nguồn. Các điều khoản của ảnh cha tiếp tục áp dụng. Danh sách 140 ảnh val và SHA của từng nhãn đã được đối chiếu sau khi đóng gói. [Sổ ảnh cha và SHA](experiments/overnight/team21-paste-manifest.json) · [Quyết định duyệt cảnh](experiments/overnight/team21-paste-scenes-accepted.json) · [Quyết định duyệt mẫu cắt](experiments/overnight/team21-paste-donors-accepted.json)

Bộ `team21-vistas-extra-v1` có 1.693 ảnh train và 140 ảnh val. Trên nền 1.442 ảnh đã QC, nhóm thêm 69 crop COCO, rồi thêm 182 cảnh Mapillary Vistas v2 thuộc split training. Phần Mapillary thêm 1.980 instance, gồm 47 xe máy và 58 xe đạp không người, 41 xe máy có người, 42 xe đạp có người, 200 đèn và 536 biển báo. Nhóm chuyển mặt nạ panoptic thành polygon, ghép người ngồi với xe và lọc theo kích thước ở ảnh gốc. Sau lượt xem trực quan, nhóm loại 24 ứng viên nghi trùng cảnh. Ảnh Mapillary sử dụng giấy phép CC BY-NC-SA 4.0. Manifest lưu URL, tác giả, giấy phép, lớp đích và SHA theo ảnh. Đây là số liệu của tập train đã chuẩn bị. Điểm mô hình được ghi sau khi chấm ZIP trên CPU. [Dữ liệu gốc](https://www.mapillary.com/dataset/vistas) · [69 crop COCO và nguồn ảnh cha](experiments/overnight/team21-context-manifest.json) · [182 cảnh Mapillary](experiments/overnight/team21-vistas-extra-manifest.json) · [Kết quả kiểm trùng](experiments/overnight/team21-vistas-extra-scenes.json)

Đội 21 dùng 50 cảnh train của bộ 1.156 ảnh để đối chiếu theo guideline BTC. Đã sửa 19 cảnh và áp dụng cùng quyết định cho ba biến thể, tổng cộng 43 thao tác. Sổ quyết định ghi cả trường hợp còn nghi vấn sau lượt tự kiểm. [Mẫu theo seed 21](experiments/overnight/team21-btc-sample.json) · [Quyết định theo cảnh](experiments/overnight/team21-btc-decisions.json) · [SHA của từng nhãn đã xuất](experiments/overnight/team21-btc-manifest.json)

Đội bổ sung 31 ảnh COCO train2017 với 59 instance sau khi rà toàn khung. Từ 38 ứng viên, sáu ảnh bị loại do cảnh có dấu hiệu trùng, một ảnh do đường biên xe máy lấn sang ô tô. Ảnh giữ giấy phép Flickr CC BY 2.0 hoặc CC BY-SA 2.0 của từng nguồn, annotation COCO dùng CC BY 4.0. Ánh xạ người đang ngồi và xe thành một instance, tách người đứng cạnh xe, chuyển bán tải sang Car, vẽ thêm từng mặt biển nhìn thấy. Bộ mới có 1.187 ảnh train và 140 ảnh val. [Manifest từng ảnh, URL nguồn, giấy phép và lớp](experiments/overnight/team21-road-manifest.json) · [Quyết định nhận và loại](experiments/overnight/team21-road-selection.json) · [Điều khoản COCO](https://cocodataset.org/#termsofuse)

Tất cả số lượng dưới đây tính theo ảnh hoặc crop đưa vào train. Validation 140 giữ nguyên ảnh và nhãn. Các bản tăng cường giữ giấy phép của ảnh cha.

Bản `e3-riderqc-v1` lấy 1.156 ảnh train của `e3-weatherqc-v1`. Tám ảnh COCO val2017 đã thuộc split train của nhóm và một ảnh mưa cùng cha được sửa nhãn theo ID annotation nguồn. Hai xích lô được chuyển sang lớp 3, người đang đi bộ được tách khỏi xe, hành khách cùng ngồi trên xe hai bánh được gộp và hình xe đạp trên mặt đường được bỏ. Toàn bộ ảnh, các nhãn khác và split được kiểm theo từng mục trong ZIP. Bản kê ghi SHA và số instance của cả 11 lớp. [COCO và điều khoản sử dụng](https://cocodataset.org/#termsofuse) · [Manifest thay đổi](experiments/overnight/coco-riderqc-manifest.json)

| Phiên bản | Thành phần bổ sung | Nguồn nhãn và ánh xạ |
|---|---|---|
| E2, 802 train | 126 COCO train2017, 32 Poribohon-BD, 4 crop biển Việt Nam v5 | COCO dùng mask instance gốc. Với Poribohon, box nguồn làm prompt SAM, duyệt lớp 3 theo loại xe và kiểm các đối tượng khác trong crop. 6 ảnh/crop Poribohon khác được khóa riêng cho holdout. |
| E3 Việt Nam, 834 train | 7 crop camera giao thông từ Vietnam Traffic Detection v2 và 25 crop từ Traffic Sign in Ho Chi Minh City v4 | Box thành prompt, xem lại đường biên và lớp. Thêm 10 Car, 22 Motorcyclist, 26 Sign. Camera ban ngày có trường hợp tên lớp trong nguồn đảo ô tô và xe máy. Metadata của hai phiên bản khai báo Apache 2.0. [Vietnam Traffic Detection](https://www.kaggle.com/datasets/huanhuyle/vietnam-traffic-detection) · [Traffic Sign in Ho Chi Minh City](https://www.kaggle.com/datasets/nhduynguyn/traffic-sign-in-ho-chi-minh-city) |
| E3 sửa taxonomy, 1.113 train | Giữ thêm 242 COCO train2017 và 37 cửa sổ KITTI instance segmentation 2015 sau duyệt | COCO lưu license Flickr riêng cho từng ảnh, annotation CC BY 4.0. KITTI ghi CC BY-NC-SA 3.0. Dùng mask gốc, ghép rider đã kiểm với xe, tách cụm biển và đèn, đối chiếu ảnh cha với split và holdout. Sổ sửa ghi checksum từng label thay đổi. [COCO](https://cocodataset.org/#termsofuse) · [KITTI](https://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015) · [Sổ sửa](experiments/overnight/taxonomy-decisions.json) |

### Bộ thử nghiệm cảnh mưa, đêm và giao thông Việt Nam

Bản `e3-weather-v1` thêm 43 ảnh/crop và 367 instance vào E3 taxonomy. Bộ mới có 1.156 ảnh train và 140 ảnh validation. SHA-256 của ZIP là `98c08b696635b776991aa26d8f0eafbbf8781c45a6bfb4c25dcbc9bbac06dbb8`. Từng ảnh và nhãn của validation khớp manifest gốc.

| Nguồn | Ảnh / instance thêm | Cách chọn và ánh xạ |
|---|---:|---|
| MUSES, ETH Zurich | 35 / 351 | Dùng split train và mask panoptic gốc của 27 chuyến quay. Gồm 14 ảnh trời quang ban ngày, 9 ban đêm, 7 mưa ban ngày, 2 mưa ban đêm và 3 sương mù ban ngày. Ánh xạ người, xe và các thành phần biển/đèn. Ghép người ngồi lên xe tương ứng. Đã xem hình có đường biên, đối chiếu raster và kiểm trùng cả ảnh cha. Ảnh, mask và crop lưu trong kho làm việc hạn chế theo EULA nghiên cứu của ETH. [Trang tác giả và giấy phép](https://muses.ethz.ch/) |
| Vietnam Traffic Vehicle Detectio v1 | 8 / 16 | Nguồn CC BY 4.0, dùng phần train của bản chọn tại revision `de7f424`. Kiểm Git blob ảnh/nhãn. SHA ảnh trong CSV của tác giả bản chọn khác cả 74 file khảo sát, vì vậy ghi lại SHA của chính dữ liệu sử dụng. Sửa polygon nguồn với đề xuất YOLO26x-seg và kiểm trực quan. Đã loại crop có người đi xe bị cắt quá nhiều và ảnh cảnh trùng nghi vấn. [Nguồn gốc](https://universe.roboflow.com/phan-ky-dat/vietnam-traffic-vehicle-detectio/dataset/1) · [Revision bản chọn](https://github.com/levienbang/yolo11/tree/de7f424f4ea205f5de6bb5243e06dd2d21a585ff/datasets/vietnam-seg-review-100) |

### BDD100K semantic và Mapillary Vistas

Hai bộ mở rộng kế tiếp chỉ bổ sung ảnh train. Đã kiểm SHA-256 của toàn bộ 140 cặp validation trước khi chuyển các ZIP lên L4.

| Bộ | Quy trình duyệt | Dữ liệu giữ lại |
|---|---|---|
| BDD100K semantic train | Quét 7.000 mask, tải 385 cặp và xem 354 cửa sổ với ảnh gốc và đường biên. Loại cụm dính nhiều xe, nhầm người và xe, hoặc nhầm biển và đèn. Trong 49 cửa sổ đạt duyệt trực quan, loại thêm 6 sau kiểm trùng cảnh. Điều khoản Berkeley áp dụng cho ảnh và mask. [Giấy phép dữ liệu](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst) | 43 crop, 139 instance. Cộng vào bộ thời tiết được 1.199 ảnh train. SHA của ZIP là `5e056fb7fead9c24f78db2f7057d77ea57743fa5a4c15aad01b6857b31f5199e`. |
| Mapillary Vistas v2.0, split train | Từ 373 cặp ảnh và mask panoptic đã xác minh, xem lần lượt 328 crop với đường biên. 238 crop qua kiểm trực quan. Loại tiếp 26 crop có dấu hiệu trùng cảnh. Dùng segment ID, ánh xạ theo tên lớp và ghép rider với xe. Ghi ROI, SHA nguồn, SHA nhãn và IoU chuyển polygon theo instance. Nguồn do Neuhold và cộng sự, Mapillary Research, phát hành theo CC BY-NC-SA 4.0. [Trang tác giả](https://www.mapillary.com/dataset/vistas) · [Bản dữ liệu dùng](https://huggingface.co/datasets/ali-sh07/Mapillary-dataset/tree/16ce879b802c3873a1a9e0da4113de0db1b522bf) · [Quyết định QC](experiments/overnight/vistas-qc.json) | 212 crop, 3.885 instance, có đủ 11 lớp. Cộng vào bộ BDD được 1.411 ảnh train. SHA của ZIP là `3cf51625f61792c51914e5f3a661f25abe4d5380f69c1f8f4c3cc3d2329afc0d`. |

### Sửa nhãn Việt Nam và tập đánh giá Mapillary riêng

Bản `e3-vnrepair-v1` giữ 1.411 ảnh train của bản Vistas. Sau khi duyệt 60 ảnh Việt Nam và 118 vùng phóng to quanh người và xe, đã sửa 24 tệp nhãn với 17 lần đổi lớp, 14 lần thêm, 9 lần xóa và 9 lần thay đường bao. So sánh từng thành phần trong ZIP xác nhận chỉ 24 tệp nhãn train thay đổi. SHA-256 của ZIP sau sửa là `03b99ac7680044dfb2ae2b1e22839d78fcfc20cf853e6c33c39b2b4d5e52daa2`. [Quyết định theo ảnh](experiments/overnight/vn-original-polygon-qc.json)

Tập chẩn đoán Mapillary lấy từ 109 ảnh cha khác với 373 ảnh nguồn đã khảo sát để train. Đã xem 91 cặp ảnh và overlay, loại các cặp có dấu hiệu trùng cảnh khi kiểm cả ảnh cha và crop. Khóa 46 ảnh trước suy luận. Tập này đo khả năng khái quát sang cảnh khác, giữ độc lập với tiêu chí chọn checkpoint trên val140. Phân bố GT thô có 10 lớp, lớp Other vehicle được đo trên holdout 6 ảnh riêng. [Manifest, giấy phép và checksum](experiments/overnight/probe-lock.json)

### Rà biển báo và người đi xe trên ảnh COCO của nhóm

Bản `e3-signqc-v1` kế thừa 1.156 ảnh train của bản riderQC. Sau khi xem 120 vùng phóng to, nhóm có bản sửa cho 11 ảnh gốc và 4 biến thể trong train. Thêm 46 dòng polygon và bỏ 2 dòng, gồm sửa liên kết người với xe đạp ở ảnh 277005. Giữ nguyên 140 cặp ảnh và nhãn validation theo SHA-256. Các ảnh khảo sát từ IDD chưa được đưa vào bộ dữ liệu này. [Quyết định sửa](experiments/overnight/focus-sign-edits.json) · [Manifest và số instance của 11 lớp](experiments/overnight/signqc-manifest.json)

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

### Rà lại nhãn trên bộ thời tiết

`e3-weatherqc-v1` kế thừa cùng 1.156 ảnh train và điều kiện sử dụng của `e3-weather-v1`. Có 13 thao tác trên 11 ảnh BDD và 49 thao tác trên 24 ảnh Việt Nam. Ví dụ, một polygon ở mép phải bám theo ô tô đằng sau hàng rào nhưng mang nhãn Sign. Một polygon khác bao cả đầu kéo và phần rơ-moóc. Hai xe tiện ích nhỏ còn khó phân biệt với pickup được ghi lại để duyệt tiếp. Đã so từng thành phần ZIP với bản cha và xác minh SHA của 140 cặp validation. [Quyết định theo ảnh](experiments/overnight/weather-original-polygon-qc.json) · [Checksum và số instance từng lớp](experiments/overnight/weatherqc-manifest.json)

## Kế hoạch nguồn của các phiên bản cũ

| STT | Tên Bộ Dữ Liệu | Đường Dẫn (URL) | Giấy Phép (License) | Số Ảnh Đã Dùng (Train / Val) | Ghi Chú Xử Lý |
| :-: | :--- | :--- | :--- | :---: | :--- |
| **1** | **COCO 2017 (Traffic Subset)** | `https://cocodataset.org/`<br>`http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | **CC BY 4.0** (Annotations) | **v1:** `320 / 80`<br>**v2:** `800 / 200` | Lọc & ánh xạ tự động bằng `scripts/prepare_coco_traffic.py`; gộp `person + motorcycle/bicycle` thành `Motorcyclist`/`Cyclist`. |
| **2** | **Dữ liệu Tổng hợp Cảnh Khó (Đêm / Mưa)** | Sinh từ tập Train COCO qua `scripts/validate_and_augment.py` | Tự tạo từ tập Train (CC BY 4.0) | **v1:** `80 / 0`<br>**v2:** `200 / 0` | Chỉ đưa vào `train`, tuyệt đối không đưa vào `val`. |
| **3** | **Dashcam / Video Đường Phố VN (Tự thu thập)** | *(Điền link YouTube / nguồn tự quay khi bổ sung ở v3)* | Tự thu thập & gán nhãn | `0 / 0` *(dự kiến v3)* | Prelabel bằng `yolo26x-seg.pt` + sửa tay trên CVAT cho `Other vehicle`, `Sign`. |
| **4** | **IDD (Indian Driving Dataset)** *(nếu dùng thêm)* | `https://idd.insaan.iiit.ac.in/` | Non-Commercial / Research (IDD License) | `0 / 0` | Lấy thêm xe máy, `autorickshaw` (`Other vehicle`), `rider`. |
| **5** | **BDD100K (Instance Seg)** *(nếu dùng thêm)* | `https://bdd-data.berkeley.edu/` | Berkeley DeepDrive dataset license; BSD 3-Clause áp dụng cho mã nguồn | `0 / 0` | Lấy thêm cảnh đêm, trời mưa. |
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
