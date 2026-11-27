# AllnAction RoboTaxi — Quy trình dự thi và nhiệm vụ của 5 thành viên

Mỗi người mở **Job mang tên mình** ở bảng dưới, rà đủ 60 ảnh và lưu các polygon đã sửa. Sau lượt duyệt chéo, đội xuất lại dữ liệu từ CVAT để huấn luyện. **Hạn nộp và chọn bài tính điểm private là 10:30 thứ Bảy, 28/11/2026, giờ Việt Nam.**

Tại thời điểm bàn giao, project có **300 ảnh, 5 Job và 4.281 polygon nháp**. Cả 5 tài khoản đã được kiểm tra quyền mở Job tương ứng. Năm ZIP trên Drive là ảnh chụp bộ dữ liệu trước khi người dùng sửa nhãn. [Project CVAT](http://34.58.69.170:8080/projects/1) · [Drive dữ liệu của đội](https://drive.google.com/drive/folders/1G06RBQwL3igPiVxMB3c-GYB_gwQvA8M0)

## 1. Ai làm gì và bàn giao cho ai?

Mỗi người phụ trách **tất cả 11 lớp** trong Job của mình. Mỗi Job có 15 ảnh ban ngày, 15 ảnh ban đêm, 15 ảnh mưa và 15 ảnh đường phố Việt Nam. Cột cuối nêu trọng tâm của lượt kiểm tra thứ hai.

| Thành viên · tài khoản | Job CVAT | Khối lượng ban đầu | Trọng tâm |
|---|---|---:|---|
| Nguyễn Anh Tiến · `anhtien` | [Task_01_AnhTien](http://34.58.69.170:8080/tasks/1/jobs/1) | 60 ảnh · 874 polygon | Ô tô con, xe buýt, xe tải bị che khuất |
| Lương Hoàng Bảo Trân · `baotran` | [Task_02_BaoTran](http://34.58.69.170:8080/tasks/2/jobs/2) | 60 ảnh · 821 polygon | Xe máy có người, xe máy trống và các xe đi sát nhau |
| Phan Hải Đăng · `haidang` | [Task_03_HaiDang](http://34.58.69.170:8080/tasks/3/jobs/3) | 60 ảnh · 876 polygon | Biển báo, đèn giao thông nhỏ, xa hoặc thiếu sáng |
| Lê Trung Sang · `trungsang` | [Task_04_TrungSang](http://34.58.69.170:8080/tasks/4/jobs/4) | 60 ảnh · 842 polygon | Ba gác, xích lô, rơ-moóc và các phương tiện thuộc lớp khác |
| Vũ Tuyết Trinh · `tuyettrinh` | [Task_05_TuyetTrinh](http://34.58.69.170:8080/tasks/5/jobs/5) | 60 ảnh · 868 polygon | Người đi bộ, xe đạp trống và người đang đạp xe |

### Nhiệm vụ ngoài Job và đầu ra của từng người

Phân công phối hợp dưới đây là đề xuất cho các khâu sau gán nhãn. Các Job trong bảng trên đã được gán trực tiếp trên CVAT.

| Người phụ trách | Việc thực hiện | Bàn giao |
|---|---|---|
| **Tiến — nguồn dữ liệu và nộp bài** | Đối chiếu ID lớp giữa nguồn, CVAT và YAML. Ghi nguồn, giấy phép, số ảnh thực dùng sau lọc. Theo dõi các mốc thi, nhận ZIP đã chấm thử từ Sang, nộp bằng tài khoản đội và chọn bài private. | Gửi Đăng danh mục nguồn và manifest ảnh. Lưu mã bài nộp, điểm public và xác nhận các bài được chọn. Phụ trách slide tổng quan và nguồn dữ liệu. |
| **Trân — dữ liệu Việt Nam và quy ước người lái** | Gom các ví dụ xe máy bị gộp hoặc tách sai. Chụp cùng một khung hình trước và sau khi sửa. Nếu vòng đánh giá cho thấy thiếu cảnh, chọn ảnh bổ sung từ nguồn có quyền sử dụng, ghi video/cảnh gốc và loại khung gần trùng. | Gửi Đăng các ca cần thống nhất nhãn, gửi Tiến danh sách ảnh bổ sung cùng nguồn. Phụ trách slide minh họa cách sửa mask. |
| **Đăng — duyệt chéo và khóa tập validation** | Điều phối việc xem lại những frame được đánh dấu. Kiểm tra biển, đèn, lớp hiếm và các ca bất đồng. Xuất bộ nhãn sau sửa, gom ảnh cùng cảnh/video và các biến thể vào cùng một split. | Gửi Trinh và Sang danh sách train/val, số ảnh mỗi split, mã băm manifest và phiên bản tập val đã khóa. Phụ trách slide quy trình duyệt và chia dữ liệu. |
| **Sang — huấn luyện, xuất ONNX và chấm thử** | Nhận bản dữ liệu đã qua kiểm tra. Chạy notebook với seed đội, ghi cấu hình và thời lượng. Xuất checkpoint tốt nhất, chạy lại ONNX bằng phần chấm thử của notebook, đối chiếu các vòng trên cùng tập val. | Gửi Tiến `submission.zip`, điểm val, phiên bản dữ liệu và tên lần chạy. Lưu `best.pt`, notebook và `run_report.json`. Phụ trách slide mô hình và kết quả. |
| **Trinh — kiểm tra dữ liệu và phân tích lỗi** | Kiểm tra cặp ảnh–nhãn, tọa độ, số đỉnh, phân bố 11 lớp và trùng ảnh giữa hai split. Cùng Sang xem các ảnh có lỗi nặng, ghi rõ lỗi do bỏ sót, nhầm lớp hay đường biên. Đề xuất một thay đổi dữ liệu cho vòng kế tiếp. | Gửi Sang kết quả kiểm tra và bản dữ liệu dùng để train. Gửi cả đội bảng lớp yếu, ví dụ lỗi và mức thay đổi điểm qua từng vòng. Phụ trách slide cải thiện dữ liệu. |

## 2. Quy trình từ nguồn ảnh đến bài nộp

### Bước 1 — Chuẩn bị công cụ và chốt quy ước

Tiến kiểm tra đường dẫn project và quyền truy cập. Máy CVAT của nhóm dùng một NVIDIA L4. Nuclio phục vụ SAM ViT-H và YOLOv7 cho việc tạo nhãn. Sang chuẩn bị môi trường train bằng notebook của đội, với mô hình dự thi **`yolo26n-seg`, ảnh đầu vào 640, `TEAM_ID = 21`**. [Notebook huấn luyện và xuất mô hình](notebooks/train_and_export.ipynb)

| ID | Lớp | ID | Lớp |
|---:|---|---:|---|
| 0 | Car | 6 | Bicycle |
| 1 | Bus | 7 | Cyclist |
| 2 | Truck | 8 | Pedestrian |
| 3 | Other vehicle | 9 | Sign |
| 4 | Motorcycle | 10 | Traffic light |
| 5 | Motorcyclist | | |

Project dùng hình học polygon và màu mặc định của CVAT. Khi xuất dữ liệu, đối chiếu **tên lớp và ID trong YAML** vì ID nội bộ của CVAT có thể khác ID huấn luyện. [Cấu hình 11 lớp](configs/traffic_11_classes.yaml)

### Bước 2 — Thu thập và tạo nhãn nháp

| Bộ dữ liệu thực dùng cho 300 ảnh CVAT | Cách lấy mẫu và tạo nhãn | Nguồn và giấy phép |
|---|---|---|
| BDD100K · 225 ảnh | 75 ảnh ban ngày, 75 ảnh ban đêm, 75 ảnh mưa. Dùng `weather` và `timeofday` để chia nhóm. Box gốc làm đầu vào cho SAM tạo mask. | Giấy phép dữ liệu Berkeley cho mục đích giáo dục, nghiên cứu và phi lợi nhuận. [Điều khoản](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst) · [Bản phân phối đã dùng](https://huggingface.co/datasets/dronefreak/BDD100K) |
| Vietnam-Traffic-Sign-Detection v5 · 75 ảnh | Chọn 75 tên khung hình gốc khác nhau. Ánh xạ các loại biển thành `Sign`. Dùng box biển gốc và phát hiện bổ sung của YOLOv7 làm đầu vào cho SAM. | Bản xuất và trang nguồn ghi Public Domain. [Roboflow Universe](https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8/dataset/5) · [Bản xuất COCO](https://www.kaggle.com/datasets/lvnduy/street-traffic-signs-in-vietnam-coco) |

Các ZIP ban đầu lưu tên ảnh, nguồn, SHA-256 và giấy phép. Khi thêm ảnh ở vòng sau, Tiến cập nhật số ảnh thực dùng và cách ánh xạ về 11 lớp. Nếu ghép với bộ COCO của vòng baseline, ghi số ảnh COCO trong phiên bản huấn luyện tương ứng. [Báo cáo nguồn dữ liệu của đội](DATASET_PROVENANCE_REPORT.md)

### Bước 3 — Sửa từng ảnh trên CVAT

1. Mở Job ở bảng phân công. Quét cả khung hình để tìm vật thể bị sót, sau đó phóng to vùng đông xe, biển báo, đèn giao thông và vùng che khuất.
2. Kiểm tra **mỗi đối tượng có một instance riêng**. Khi người đang lái xe máy hoặc đạp xe, vẽ một polygon cho cả người và xe, lần lượt chọn `Motorcyclist` hoặc `Cyclist`. Xe trống dùng `Motorcycle` hoặc `Bicycle`. Người đi bộ cạnh xe được gán riêng nếu nhìn thấy rõ.
3. Chỉnh đường biên theo phần vật thể nhìn thấy. Tách hai xe đi sát nhau, xóa nhãn trùng và bổ sung đối tượng thiếu. Cả nhóm tìm `Other vehicle` vì lớp này có **0 polygon trong bản nháp**. Khi gặp trường hợp khó, ghi số frame và câu hỏi vào issue của Job.
4. Lưu trước khi rời Job. Sau khi rà đủ 60 ảnh, báo cho Đăng số frame còn vướng và ví dụ trước/sau. Chuyển Job sang `Completed` khi đã lưu đủ và xử lý các điểm cần sửa.

### Bước 4 — Duyệt chéo, xuất dữ liệu và khóa tập val

Đăng cùng người phụ trách xem lại các frame có issue và chọn mẫu từ cả bốn nhóm cảnh ở mỗi Job. Nếu thành viên chỉ có quyền mở Job của mình, chủ Job chia sẻ màn hình để người duyệt đối chiếu. Đăng ghi kết luận cùng số frame để người gán nhãn sửa và lưu.

Sau khi duyệt, Đăng xuất project từ CVAT bằng định dạng **Ultralytics YOLO Segmentation**, kèm ảnh. Nếu cần quyền xuất, Tiến thao tác bằng tài khoản quản trị. Lưu ZIP mới theo phiên bản trên Drive. Trinh mở thử ảnh và polygon trong bản xuất để đối chiếu với frame đã sửa trên CVAT.

Đăng chia train/val theo **video hoặc cảnh quay**. Ảnh gần nhau, ảnh gốc và mọi bản tăng cường của nó thuộc cùng một split. Năm Job là cách chia việc cho người gán nhãn. Việc chọn val cần dựa trên nguồn/cảnh. Ghi danh sách ảnh và mã băm của val, sau đó dùng cùng phiên bản này cho các vòng so sánh.

### Bước 5 — Kiểm tra dữ liệu và huấn luyện

Trinh kiểm tra cấu trúc `images/train`, `images/val`, `labels/train`, `labels/val`. Mỗi ảnh có file nhãn cùng tên gốc. Mỗi dòng YOLO-seg có dạng `class_id x1 y1 x2 y2 x3 y3 ...`, ít nhất ba đỉnh khác nhau, tọa độ hữu hạn trong `[0,1]`. Mở lại ảnh có file nhãn trống, polygon tự cắt hoặc mask nhỏ để quyết định theo vật thể thực. Kiểm tra ảnh trùng và gần trùng giữa hai split bằng manifest và công cụ trong notebook.

Từ thư mục gốc repo, chạy lệnh sau với đường dẫn giải nén thực tế. Script in số ảnh, dòng nhãn lỗi, mask nhỏ và phân bố lớp cho từng split. Sửa lỗi rồi chạy lại. [Script kiểm tra dữ liệu](scripts/validate_and_augment.py)

```bash
python scripts/validate_and_augment.py --data /duong/dan/bo_du_lieu_da_duyet
```

Sang mở notebook, đặt `DATASET_PATH` tới **bản vừa duyệt** và `DRIVE_OUTPUT_DIR` tới thư mục riêng của lần chạy. Ô dựng COCO tự động phục vụ luồng COCO trong notebook; khi dùng bản CVAT đã chuẩn bị, nạp trực tiếp đường dẫn đó. Chạy các bước nạp dữ liệu, kiểm tra rồi huấn luyện. Lưu commit, phiên bản dữ liệu, seed, tham số, thời lượng và checkpoint. [Notebook của đội](notebooks/train_and_export.ipynb)

### Bước 6 — Phân tích lỗi và cải thiện

Sang xuất ONNX rồi cùng Trinh tính **điểm val = 100 × mask mAP@[.5:.95]**. Xem AP theo lớp và một số ảnh có dự đoán sai nhiều. Chọn một lỗi cụ thể, ví dụ xe máy có người bị tách thành hai instance. Trân và người phụ trách sửa các ví dụ train liên quan, Đăng duyệt, Sang chạy lại với cùng val. Ghi cả vòng có điểm giảm để giải thích quyết định chọn checkpoint.

Repo ghi baseline **21,92 trên tập val 80 ảnh của vòng v1**. Nếu dùng một tập val mới cho bộ CVAT, hãy chạy checkpoint baseline trên chính tập mới trước khi tính mức cải thiện. Ghi riêng điểm val, public và private. [Báo cáo baseline](REPORT_V1_BASELINE.md) · [Nhật ký thí nghiệm](EXPERIMENTS.md)

### Bước 7 — Kiểm tra gói, nộp và chọn bài private

1. Sang dùng notebook mẫu để xuất ONNX **FP32** từ `yolo26n-seg` ở kích thước 640. Chạy bước chấm thử bằng ONNX Runtime CPU và kiểm tra đầu ra mask. Gói nộp có tên `submission.zip`, tối đa **42 MB** theo đề, chứa đúng **một file `model.onnx` ở thư mục gốc của ZIP**.
2. Sang gửi Tiến ZIP đã thử, điểm val và mã phiên bản. Lưu riêng `best.pt`, notebook đã chạy, ZIP dữ liệu, manifest và `run_report.json` để tái lập. Tiến ghi mã bài, thời điểm nộp, trạng thái chấm và điểm public vào nhật ký đội.
3. Đội có **20 lượt nộp**, mỗi lần có tối đa **một bài đang chấm**. Dành thời gian cho hàng đợi và xử lý lỗi trước hạn. Tiến chọn tối đa **hai bài đã chấm** để tính private và lưu xác nhận lựa chọn trước **10:30 ngày 28/11**. Nếu đội bỏ qua bước chọn, hệ thống lấy hai bài đã chấm gần nhất.

### Bước 8 — Bàn giao báo cáo và thuyết trình

Mỗi người chuyển cho Tiến phần slide trong bảng phân công. Ghép theo trình tự **nguồn và giấy phép → ví dụ nhãn trước/sau → duyệt và chia tập → baseline và vòng cải thiện → điểm, lỗi còn lại và cách tái lập**. Mỗi con số trên slide ghi rõ tập đánh giá và phiên bản. Chuẩn bị sẵn một ảnh sửa nhãn thực tế, một ví dụ mô hình còn sai và notebook của bài đã chọn.

| Mốc chính thức · giờ Việt Nam (UTC+7) | Đầu việc của đội |
|---|---|
| 09:00 ngày 27/11/2026 | Mở nộp bài. Đội có thể nộp sớm một bản hợp lệ đã chấm thử. |
| 10:30 ngày 28/11/2026 | Đóng nộp. Hoàn tất chọn bài private trước thời điểm này. |
| 11:00 ngày 28/11/2026 | Công bố bảng private. Tiến cập nhật kết quả thực tế vào slide. |
| 11:30 ngày 28/11/2026 | Bắt đầu thuyết trình đối với các đội được chọn. Cả nhóm sẵn sàng mở lại dữ liệu, notebook và báo cáo lần chạy. |

Xếp hạng cuối kết hợp **50% thứ hạng theo điểm private và 50% đánh giá giải pháp**. Khi bàn giao lần cuối, Tiến kiểm tra nhóm mở được bản dữ liệu sau duyệt, notebook, `run_report.json`, gói ONNX của từng bài đã chọn và slide của đội.
