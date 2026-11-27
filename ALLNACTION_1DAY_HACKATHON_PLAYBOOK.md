# AllnAction Robotaxi Hackathon — Kế Hoạch 1 Ngày & Tổ Chức GitHub (5 Thành Viên)

- **Bài toán:** Traffic Instance Segmentation (**11 lớp giao thông**)
- **Mô hình:** `yolo26n-seg`
- **Thước đo chấm điểm:** **Mask mAP@[.5:.95]** (`metrics/mAP50-95(M)`)
- **Hình thức:** 5 thành viên làm **online song song**, train trên **Colab / Kaggle GPU**, dữ liệu tự xây dựng trong **1 ngày**.

---

## 1. Tổ Chức GitHub & Lưu Trữ Dữ Liệu

### 1.1. Cấu Trúc Repo GitHub
**Nguyên tắc:** GitHub chỉ chứa code, config và script. **Tuyệt đối không push ảnh, file nhãn `.txt` nặng hoặc file `.pt` lên GitHub.**

```text
allnaction-traffic-seg/
├── .gitignore                  # Chặn dataset/, runs/, *.pt, *.zip, *.jpg, *.png
├── README.md                   # Hướng dẫn chạy nhanh cho cả đội
├── requirements.txt            # ultralytics, pillow, numpy, pandas, scipy
├── configs/
│   └── traffic_11_classes.yaml # File YAML chuẩn 11 lớp dùng chung toàn đội
├── scripts/
│   ├── seed_utils.py           # Khóa TEAM_SEED = 2026
│   ├── auto_prelabel.py        # (Người 5) Chạy YOLO/SAM tạo nhãn nháp
│   ├── validate_dataset.py     # (Người 5) Kiểm tra lỗi polygon, tọa độ [0,1], leak train/val
│   ├── copy_paste_aug.py       # (Người 5) Nhân bản mask lớp hiếm
│   └── train_eval.py           # (Người 4) Train yolo26n-seg & chấm Mask mAP@[.5:.95]
└── notebooks/
    └── colab_runner.ipynb      # Notebook 1-click kéo Git + unzip data + train trên Colab/Kaggle
```

### 1.2. Nội Dung `.gitignore` Bắt Buộc
```gitignore
dataset_traffic_11/
data/
runs/
*.pt
*.onnx
*.zip
*.jpg
*.jpeg
*.png
__pycache__/
.ipynb_checkpoints/
```

### 1.3. Quy Tắc Branch GitHub (Tránh Conflict)
- `main`: Code chạy ổn định đã test trên Colab.
- `feat/data-tools`: Nhánh riêng của **Người 5** (pre-label, validate, copy-paste, eval).
- `feat/train-pipeline`: Nhánh riêng của **Người 4** (train, tune hyperparameter, TTA).
- Người 1, 2, 3 không cần sửa code; chỉ pull `main` nếu cần chạy script local.

### 1.4. Nơi Lưu Data & Weights
| Thành Phần | Nơi Lưu Trữ | Quy Ước Đặt Tên |
| :--- | :--- | :--- |
| **Gán nhãn online** | **CVAT Online / Roboflow** | Chung 1 workspace, 11 lớp cố định (`0..10`). |
| **Kho Data trung tâm** | **Google Drive chung (`/AllnAction_Data/`)** | `val_locked.zip`, `shardA_v1.zip`, `shardB_v1.zip`, `shardC_v1.zip`. |
| **Weights & Logs** | **Google Drive (`/AllnAction_Runs/`)** | Lưu `best.pt`, `results.csv`, biểu đồ PR/Confusion Matrix. |

---

## 2. Phân Công 5 Thành Viên Chạy Song Song 100%

| Thành Viên | Vai Trò | Nhiệm Vụ Cụ Thể | Đầu Ra |
| :--- | :--- | :--- | :--- |
| **Người 1** | **Data Shard A (Cao tốc / Ban ngày)** | Gom ảnh & sửa nhãn pre-label cho cảnh đường thoáng, xe lớn (`car`, `truck`, `bus`, `road`). | `shardA_XXXX.jpg` + `.txt` |
| **Người 2** | **Data Shard B (Đô thị / Giao lộ)** | Gom ảnh & sửa nhãn cảnh đông (`pedestrian`, `motorcycle`, `bicycle`, `traffic_light`, `traffic_sign`). | `shardB_XXXX.jpg` + `.txt` |
| **Người 3** | **Data Shard C (Đêm / Mưa / Edge-case)** | Gom ảnh & sửa nhãn cảnh khó (`obstacle_cone`, `lane_marking` mờ, ánh sáng yếu). | `shardC_XXXX.jpg` + `.txt` |
| **Người 4** | **GPU & Training Lead** | Dựng Colab/Kaggle GPU, khóa `TEAM_SEED=2026`, chạy smoke-test bằng data giả lập, train & tune `yolo26n-seg`. | `best.pt` + Bảng điểm `mAP@[.5:.95]` |
| **Người 5** | **Auto-Prelabel, QC & Eval Lead** | Map open dataset, chạy auto-prelabel hỗ trợ 1-2-3, viết `validate_dataset.py`, làm `copy_paste`, soi lỗi từng lớp. | `val_locked.zip` + `data_v1/v2.zip` + Báo cáo lỗi |

**Quy tắc prefix chống trùng file:** Mọi ảnh và nhãn của Shard A, B, C bắt buộc đặt tiền tố `shardA_`, `shardB_`, `shardC_`. Khi gộp chỉ cần giải nén vào chung `images/train` và `labels/train`.

---

## 3. Chiến Lược Gán Nhãn 1 Ngày (Không Vẽ Tay Từ Đầu)

1. **Bước 1 — Tận dụng Open Dataset (30 phút):** Kéo một phần ảnh từ BDD100K / Cityscapes / Roboflow Universe cho 8–9 lớp phổ thông, chạy script map ID về `0..10`.
2. **Bước 2 — Auto-Prelabel:** Chạy `yolo26n-seg.pt` (hoặc model COCO lớn) để tự sinh file `.txt` nháp cho xe/người/đèn; dùng **SAM** trên CVAT/Roboflow click 1 lần cho `road`, `lane_marking`, `obstacle_cone`.
3. **Bước 3 — Review & Fix:** Người 1, 2, 3 chỉ chỉnh sửa biên mask bị lẹm và bổ sung vật thể nhỏ bị sót (~1 phút/ảnh).
4. **Mục tiêu số lượng trong 1 ngày:**
   - **Tập `val` (khóa cứng):** **80 – 120 ảnh** sạch tuyệt đối, đủ 11 lớp.
   - **Tập `train`:** **300 – 600 ảnh** (ưu tiên biên mask ôm sát để giữ IoU cao ở ngưỡng `0.75–0.95`).

---

## 4. Lộ Trình Chi Tiết Trong 1 Ngày (8 Giờ Thực Chiến)

| Khung Giờ | Người 1, 2, 3 (Data Shards) | Người 4 (GPU & Model) | Người 5 (Tooling & Eval) |
| :--- | :--- | :--- | :--- |
| **Giờ 0 – 1**<br>*(Khởi động)* | Gom ảnh thô theo 3 bối cảnh A, B, C. | Tạo repo GitHub, mở Colab/Kaggle GPU, test chạy smoke-test `yolo26n-seg`. | Chạy Auto-Prelabel, map open dataset, chốt xong **`val_locked.zip` (100 ảnh)**. |
| **Giờ 1 – 3**<br>*(Data v1)* | Sửa nhãn nhanh, mỗi người xuất **80–100 ảnh** → đẩy `shardX_v1.zip` lên Drive. | Tune sẵn `colab_runner.ipynb`, chuẩn bị script train `imgsz=640`, `mask_ratio=2`. | Viết & chạy `validate_dataset.py` lọc polygon lỗi, gộp thành **`data_v1.zip`**. |
| **Giờ 3 – 5**<br>*(Train v1 + Data v2)* | Tiếp tục làm **Data v2 (+100 ảnh/người)** trong lúc GPU đang chạy. | Giải nén `data_v1.zip` vào `/content/`, chạy train **Baseline v1** (`100 epochs`). | Phân tích kết quả v1 (`metrics.seg.maps`), chỉ ra 3 lớp có `mAP` thấp nhất. |
| **Giờ 5 – 7**<br>*(Data v2 + Ép điểm)* | Tập trung bổ sung đúng 3 lớp yếu nhất + hoàn tất `shardX_v2.zip`. | Train **Model v2** với `data_v2.zip`, tăng `imgsz=960`, `copy_paste=0.3`, `AdamW`. | Chạy `copy_paste_aug.py` nhân bản lớp hiếm (`cone`, `bicycle`, `traffic_light`). |
| **Giờ 7 – 8**<br>*(Chốt hạ)* | Kiểm tra lần cuối các mẫu sai điển hình trên ảnh test. | Chạy `model.val()` với `iou=0.70`, thử TTA (`augment=True`), lưu `best.pt` lên Drive. | Tổng hợp bảng điểm **Mask mAP@[.5:.95]** toàn bộ 11 lớp & file kết quả cuối. |

---

## 5. Lệnh Chạy Chuẩn Trên Colab / Kaggle GPU

```bash
# 1. Mount Google Drive (chỉ trên Colab) & kéo code từ GitHub
git clone https://github.com/<team>/allnaction-traffic-seg.git
cd allnaction-traffic-seg
pip install -U ultralytics pillow numpy

# 2. Giải nén data vào ổ SSD cục bộ /content/ (KHÔNG train trực tiếp trên Drive)
mkdir -p /content/dataset_traffic_11/images/train /content/dataset_traffic_11/labels/train
mkdir -p /content/dataset_traffic_11/images/val   /content/dataset_traffic_11/labels/val

unzip -q "/content/drive/MyDrive/AllnAction_Data/val_locked.zip" -d /content/dataset_traffic_11/
unzip -q "/content/drive/MyDrive/AllnAction_Data/shard*_v2.zip"  -d /content/dataset_traffic_11/

# 3. Kiểm tra lỗi nhãn & chạy train + chấm Mask mAP@[.5:.95]
python scripts/validate_dataset.py --data /content/dataset_traffic_11
python scripts/train_eval.py
```

---

## 6. Checklist Các Thông Số Quyết Định Điểm Mask mAP@[.5:.95]
- [ ] `TEAM_SEED = 2026` và `deterministic=True` đã khóa ở mọi lần chạy.
- [ ] Tập `val` được khóa cứng, không trộn lẫn ảnh `train` vào `val`.
- [ ] `mask_ratio=2` (thay vì mặc định `4`) để giữ độ sắc nét của biên mask.
- [ ] `imgsz=960` (hoặc `1280` nếu VRAM cho phép) để bắt tốt `traffic_light`, `traffic_sign`, `pedestrian`.
- [ ] `copy_paste=0.3` bật trong lúc train để cứu các lớp ít mẫu.
- [ ] `conf=0.001, iou=0.70` khi chạy `model.val()` để chấm đúng chuẩn **Mask mAP@[.5:.95]**.
