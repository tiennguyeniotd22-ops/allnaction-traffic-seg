# Kết quả đội 21 trên 140 ảnh validation

**Bản có mask mAP cao nhất là `team21-mask-roi`, đạt 27,54/100.** So với pilot, chênh 3,41 điểm. Mỗi dòng dưới đây đã chấm lại từ ZIP trên CPU. Điểm public và private do hệ thống cuộc thi xác định.

| Bản | Mask mAP | AP50 | AP75 | Việt Nam · 15 | Mapillary · 46 | Other vehicle · holdout |
|---|---:|---:|---:|---:|---:|---:|
| `team21-mask-roi` | 27,54 | 45,24 | 29,18 | 22,39 | 26,78 | 61,62 |
| `team21-wheel-contours` | 27,11 | 44,72 | 28,88 | 21,99 | 26,35 | 59,10 |
| `team21-mean-paste-bn` | 25,80 | 44,21 | 26,07 | 19,24 | 26,54 | 60,71 |
| `team21-mean-wide-bn` | 25,64 | 43,87 | 25,42 | 18,78 | 24,99 | 61,61 |
| `team21-mean-bn-input` | 25,60 | 43,94 | 26,20 | 19,00 | 24,98 | 62,74 |
| `team21-mean-qc-o2m` | 25,58 | 44,22 | 25,37 | 19,53 | 25,52 | 59,10 |
| `team21-mean-wide-o2m` | 25,53 | 43,91 | 25,27 | 19,01 | 25,07 | 62,31 |
| `team21-mean-sgd-bn` | 25,51 | 43,84 | 26,10 | 18,96 | 24,94 | 62,74 |
| `last-team21-polish12` | 25,28 | 43,94 | 25,61 | 18,30 | 26,14 | 52,77 |
| `team21-polish12` | 25,25 | 43,81 | 25,39 | 17,98 | 25,27 | 66,62 |
| `team21-wheel-head` | 25,25 | 40,84 | 26,52 | 19,92 | 25,61 | 61,62 |
| `team21-prior25` | 25,19 | 43,24 | 25,71 | 19,82 | 24,68 | 60,33 |
| `team21-wheel12` | 25,16 | 43,24 | 25,55 | 17,05 | 26,95 | 60,83 |
| `team21-mean-bn` | 25,15 | 43,71 | 25,47 | 18,65 | 24,42 | 51,74 |
| `team21-mean-sgd` | 25,15 | 43,42 | 25,83 | 18,70 | 24,75 | 60,76 |
| `team21-mean-four` | 24,93 | 43,35 | 25,37 | 17,85 | 24,38 | 51,29 |
| `team21-mean-road` | 24,67 | 42,42 | 24,74 | 17,90 | 24,53 | 58,73 |
| `team21-mean-bn075` | 24,64 | 42,32 | 25,22 | 18,63 | 24,46 | 56,16 |
| `last-team21-wideqc20` | 24,43 | 42,27 | 24,66 | 20,63 | 26,47 | 65,45 |
| `team21-dense-o2o` | 24,43 | 42,67 | 24,23 | 19,27 | 24,49 | 50,05 |
| `team21-balanced20` | 24,42 | 42,44 | 24,13 | 19,51 | 25,14 | 58,09 |
| `team21-wideqc20` | 24,39 | 42,26 | 24,31 | 20,84 | 25,85 | 65,45 |
| `last-team21-vnwide20` | 24,35 | 42,70 | 24,35 | 18,01 | 24,78 | 63,23 |
| `last-team21-sgd20` | 24,22 | 40,97 | 25,06 | 17,70 | 24,47 | 43,48 |
| `last-team21-paste20` | 24,20 | 42,62 | 23,98 | 18,56 | 26,77 | 56,38 |
| `team21-mean-a` | 24,18 | 41,76 | 24,15 | 18,70 | 24,65 | 54,76 |
| `team21-paste20` | 24,17 | 42,58 | 24,34 | 18,67 | 26,45 | 53,54 |
| `team21-vnwide20` | 24,17 | 42,65 | 23,77 | 18,04 | 24,68 | 62,28 |
| `team21-sgd20` | 24,14 | 40,97 | 24,49 | 17,49 | 24,56 | 43,76 |
| `team21-qc20` | 24,13 | 41,55 | 24,49 | 19,09 | 23,59 | 50,98 |
| `team21-head20` | 23,88 | 40,78 | 24,44 | 17,73 | 24,55 | 55,89 |
| `team21-vistas20` | 23,79 | 41,97 | 23,88 | 19,02 | 26,12 | 63,24 |
| `team21-btc20` | 23,62 | 40,33 | 22,91 | 16,56 | 24,06 | 61,11 |
| `team21-o2m20` | 23,48 | 41,21 | 22,83 | 17,37 | 24,17 | 51,24 |
| `team21-road20` | 23,44 | 40,71 | 23,86 | 17,45 | 23,40 | 61,57 |
| `team21-distill20` | 23,36 | 41,47 | 23,65 | 18,63 | 26,09 | 65,83 |
| `team21-full` | 23,22 | 40,66 | 23,42 | 17,12 | 22,92 | 58,10 |
| `team21-prior50` | 22,70 | 39,12 | 23,20 | 17,45 | 22,15 | 51,93 |
| `team21-quality-mask` | 22,66 | 38,03 | 23,19 | 13,32 | 22,54 | 59,01 |

`team21-btc20` so với `team21-qc20` chênh -0,51 điểm. Khoảng bootstrap ghép cặp 95% từ -1,79 đến 0,79. Cấu hình train, trọng số khởi tạo và notebook trùng nhau.

`team21-road20` so với `team21-btc20` chênh -0,18 điểm. Khoảng bootstrap ghép cặp 95% từ -1,16 đến 0,75. Cấu hình train, trọng số khởi tạo và notebook trùng nhau.

Khoảng bootstrap 95% theo nhóm cảnh của bản được chọn so với `team21-mean-paste-bn` là [1,31, 2,20] điểm (300 lần lấy mẫu). Nhiều bản được so trên cùng validation. Khoảng này chưa điều chỉnh theo số lần chọn mô hình.

So trực tiếp với ZIP đầu vòng `team21-mean-paste-bn` (25,80), chênh lệch là 1,74 điểm. Khoảng bootstrap 95% [1,31, 2,20].

| Lớp | GT hợp lệ | AP pilot | AP bản chọn |
|---|---:|---:|---:|
| Car | 546 | 40,27 | 44,93 |
| Bus | 35 | 39,52 | 43,42 |
| Truck | 82 | 19,91 | 23,82 |
| Other vehicle | 0 | Thiếu GT | Thiếu GT |
| Motorcycle | 54 | 7,45 | 9,68 |
| Motorcyclist | 70 | 35,07 | 36,24 |
| Bicycle | 44 | 13,45 | 15,34 |
| Cyclist | 33 | 15,66 | 16,58 |
| Pedestrian | 418 | 27,10 | 31,05 |
| Sign | 152 | 27,26 | 33,67 |
| Traffic light | 154 | 15,58 | 20,63 |

## Dữ liệu và trọng số

Bộ val gồm 45 ảnh BDD, 15 ảnh Việt Nam và 80 ảnh Drive. Holdout riêng của Other vehicle có 7 instance trên 6 ảnh. Mỗi lượt dùng seed 21, YOLO26n-seg, cạnh 640 và bước đóng gói từ notebook.

Đối chứng A sửa 22 tệp nhãn trên bộ 1.156 ảnh. A+B thêm 31 ảnh COCO với 59 instance. Bản head20 thử chuyển trọng số COCO sang bốn lớp đích bổ sung. Các bản có tên mean dùng trung bình trọng số FP32 của các checkpoint trước khi xuất một model ONNX FP32.

| Checkpoint đầu vào của bản chọn | Bộ dữ liệu | Epoch | Công thức |
|---|---|---:|---|
| team21-qc20 | e3-triage-v1 | 20 | lower-lr |
| team21-btc20 | team21-btc-v1 | 20 | lower-lr |
| team21-road20 | team21-road-v1 | 20 | lower-lr |
| team21-full | team21-signfaces-v1 | 55 | lower-lr |
| team21-sgd20 | e3-triage-v1 | 20 | sgd |
| team21-vnwide20 | e3-vnrepair-v1 | 20 | lower-lr |
| team21-wideqc20 | team21-wideqc-v1 | 20 | lower-lr |
| team21-vistas20 | team21-vistas-extra-v1 | 20 | lower-lr |
| team21-paste20 | team21-paste-v1 | 20 | lower-lr |

Kiểm SHA xác nhận 140 ảnh val gốc trên từng bộ. Mỗi bộ train nguồn chứa từ 640 đến 640 trong 640 ảnh train vòng 4.

Bản chọn tinh chỉnh từ `team21-mean-paste-bn` trên nhãn train. Sổ checkpoint lưu các bước và SHA trọng số cha.

Bản chọn có hiệu chỉnh BatchNorm từ 168 ảnh train thuộc ba nguồn, batch 8 và momentum 0.025.

Lượt `team21-vnwide20` dùng 1.411 ảnh train, gồm 212 crop Mapillary và 43 crop BDD. Có 24 tệp nhãn Việt Nam đã sửa. So với pilot, điểm val140 chênh 0,04 và điểm Việt Nam chênh -1,05. Hai tập train khác nhau cả về nguồn ảnh và nhãn.

Trên cùng bộ train của pilot, `team21-o2m20` giữ trọng số loss của nhánh one-to-many ở 0,8 trong 20 epoch. Pilot giảm dần từ 0,8 xuống 0,1. Chênh lệch trên val140 là -0,64 điểm, khoảng bootstrap 95% [-1,71, 0,51]. Hai lượt cùng dùng bước xuất ONNX của BTC.

Bộ 1.442 ảnh giữ các nhãn đã QC trên 48 tệp chung có khác biệt giữa hai bộ, cùng 31 ảnh COCO đã duyệt và 255 crop BDD/Mapillary. So cùng công thức train với bộ 1.411 ảnh, chênh lệch là 0,23 điểm val140, khoảng bootstrap 95% [-1,11, 1,24].

Trên máy đã có quyền đọc thư mục dự án trên VM, lấy ZIP bằng Google Cloud CLI. Thực hiện lệnh dữ liệu cho từng phiên bản cần chạy notebook.

```powershell
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-mask-roi/submission/submission.zip" . --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/e3-triage-v1/dataset.zip" "./e3-triage-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-btc-v1/dataset.zip" "./team21-btc-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-road-v1/dataset.zip" "./team21-road-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-signfaces-v1/dataset.zip" "./team21-signfaces-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/e3-vnrepair-v1/dataset.zip" "./e3-vnrepair-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-wideqc-v1/dataset.zip" "./team21-wideqc-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-vistas-extra-v1/dataset.zip" "./team21-vistas-extra-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/team21-paste-v1/dataset.zip" "./team21-paste-v1.zip" --project=plantcare-ai-510902 --zone=us-central1-a
```

Tệp nộp có 9,674,663 byte, chỉ chứa `model.onnx`. SHA-256 là `2717f063091799facee8526bab18e4fc686424109d37d1b778588e4c42b3070e`.

[Hồ sơ chạy lại và SHA dữ liệu](replay/team21-mask-roi/REPLAY.md) · [run_report.json](replay/team21-mask-roi/run_report.json) · [ZIP ONNX điểm cao nhất](../artifacts/team21-mask-roi/submission.zip) · [Nguồn dữ liệu và quyền sử dụng](../DATASET_PROVENANCE_REPORT.md) · [Notebook đã dùng](../notebooks/archive/train_and_export_v2.ipynb)
