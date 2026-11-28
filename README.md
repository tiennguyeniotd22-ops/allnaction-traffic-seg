# AllnAction RoboTaxi Data-Centric Hackathon (11 Traffic Classes)

- **Mô hình cố định:** `yolo26n-seg` (`ultralytics==8.4.214`, `imgsz=640`, xuất `ONNX` opset 17, FP32)
- **Thước đo:** `100 × Mask mAP@[.5:.95]`

---

## Cấu Trúc Repository

| Thư Mục / File | Nội Dung |
| :--- | :--- |
| [`notebooks/train_round4.ipynb`](notebooks/train_round4.ipynb) | **Notebook huấn luyện & xuất ONNX mới nhất (v4)** |
| [`notebooks/prelabel/`](notebooks/prelabel/) | Notebook auto-prelabel bằng **SAM 3** và **YOLO-World + SAM 2.1 (không cần token)** |
| [`notebooks/archive/`](notebooks/archive/) | Lưu trữ các bản cũ (`train_and_export_v2.ipynb`, `train_round3.ipynb`) để đối chiếu tái lập |
| [`configs/traffic_11_classes.yaml`](configs/traffic_11_classes.yaml) | Định nghĩa chuẩn 11 lớp giao thông (`0..10`) |
| [`DATASET_PROVENANCE_REPORT.md`](DATASET_PROVENANCE_REPORT.md) | Khai báo nguồn dữ liệu công khai, giấy phép & ánh xạ 11 lớp |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Nhật ký điểm số và đối chiếu các vòng chạy |
| [`docs/`](docs/) | Quy trình đội, hướng dẫn review CVAT (`HUONG_DAN_QC.md`) & guideline BTC |
| [`reports/`](reports/) | Báo cáo baseline v1, phân tích lỗi & đề xuất tăng điểm (`team21-qc20`) |
| [`scripts/`](scripts/) | Bộ script chuẩn bị dữ liệu, tải ảnh VN, copy-paste lớp hiếm & kiểm tra nhãn |

---

## Chạy Nhanh Trên Colab
- **Huấn luyện & xuất `submission.zip` (Bản 4 mới nhất):**
  `https://colab.research.google.com/github/tiennguyeniotd22-ops/allnaction-traffic-seg/blob/main/notebooks/train_round4.ipynb`
- **Prelabel ảnh mới bằng SAM 3 / SAM 2.1:**
  Mở `notebooks/prelabel/sam3_prelabel_11_classes.ipynb` hoặc `notebooks/prelabel/sam2_yoloworld_prelabel_notoken.ipynb`.
