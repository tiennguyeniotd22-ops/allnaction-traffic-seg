# AllnAction RoboTaxi Data-Centric Hackathon (11 Traffic Classes)

- **Mô hình cố định:** `yolo26n-seg` (`ultralytics==8.4.214`, `imgsz=640`, xuất `ONNX` opset 17)
- **Thước đo:** `100 × Mask mAP@[.5:.95]`
- **Chiến lược:** Khai thác Open Prelabel (`IDD`/`BDD100K`/`Roboflow`) → Teacher `yolo26x-seg.pt` Pseudo-Label ảnh Dashcam VN → Sửa nhanh trên CVAT → Fine-tune & Data-Centric Loop.

## Cấu Trúc Repo

Quy ước gán nhãn của giải nằm trong bản gốc do ban tổ chức cung cấp. Nhóm dùng checklist QC để duyệt polygon trên CVAT và kiểm lại mask sau khi xuất YOLO-seg. [Guideline chính thức](labeling_guideline.md) · [Checklist QC](HUONG_DAN_QC.md) · [Đề xuất tăng điểm mask](DE_XUAT_TANG_DIEM_MASK.md)

- `ALLNACTION_1DAY_HACKATHON_PLAYBOOK.md`: Kế hoạch 1 ngày & phân công 5 thành viên song song.
- `DATASET_PROVENANCE_REPORT.md`: Bảng khai báo nguồn dữ liệu công khai, giấy phép & ánh xạ 11 lớp.
- `configs/traffic_11_classes.yaml`: Định nghĩa chuẩn 11 lớp (`0..10`).
- `scripts/prelabel_yolo_seg.py`: Auto-prelabel chuẩn YOLO Segmentation 1.0 + tự gộp `Motorcyclist` / `Cyclist`.
- `scripts/validate_and_augment.py`: Kiểm tra lỗi polygon, mask `< 64px` & sinh ảnh cảnh khó (đêm/mưa).
- `notebooks/train_and_export.ipynb`: Notebook huấn luyện, xuất ONNX và tự chấm thử của cuộc thi.

## Chạy Nhanh Trên Colab / Server GPU
```bash
pip install -r requirements.txt
python scripts/prelabel_yolo_seg.py --images raw_images/ --out dataset_traffic_11 --model yolo26x-seg.pt
python scripts/validate_and_augment.py --data dataset_traffic_11 --augment-hard 0.30
```
Mở `notebooks/train_and_export.ipynb`, điền đúng `TEAM_ID` và `DATASET_PATH`, rồi chạy từ Bước 0 đến Bước 6.
