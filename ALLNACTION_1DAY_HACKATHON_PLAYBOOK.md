# AllnAction RoboTaxi Data-Centric Hackathon — Kế Hoạch & Chiến Thuật 1 Ngày (5 Thành Viên)

- **Bài toán:** Instance Segmentation giao thông Việt Nam (**11 lớp cố định**).
- **Bảng 11 lớp chuẩn:**
  `0: Car` | `1: Bus` | `2: Truck` | `3: Other vehicle` *(xích lô, xe ba gác...)* | `4: Motorcycle` *(không người)* | `5: Motorcyclist` *(xe máy + người)* | `6: Bicycle` *(không người)* | `7: Cyclist` *(xe đạp + người)* | `8: Pedestrian` | `9: Sign` *(biển báo)* | `10: Traffic light` *(đèn giao thông)*.
- **Mô hình nộp bài:** `yolo26n-seg` (`Ultralytics 8.4.214`, `imgsz=640`, xuất `ONNX` opset 17) — **Mô hình cố định, cuộc thi Data-Centric (cải thiện dữ liệu)**.
- **Thước đo:** `100 × Mask mAP@[.5:.95]`
- **Chiến lược cốt lõi:** **Khai thác Prelabel có sẵn → Dựng Model Pretrain/Teacher → Pseudo-Label ảnh Việt Nam → Fine-tune `yolo26n-seg` → Vòng lặp Data-Centric.**

---

## 1. Kiến Trúc Chiến Thuật 4 Tầng (Từ Prelabel Đến Fine-Tune)

```text
[Tầng 1: Open Prelabel]   IDD + BDD100K + Roboflow VN ──(Remap 11 lớp)──► Data Nền (60%)
                                                                              │
[Tầng 2: Model Teacher]   Train Model Pretrain / Dùng yolo26x-seg.pt ◄────────┘
                                  │
                                  ▼ (Pseudo-Label + Rule tách Motorcyclist/Cyclist)
[Tầng 3: Data Việt Nam]   Ảnh Dashcam VN ──► Sửa nhanh trên CVAT ──► Tập Val Khóa Cứng + Train VN (30%)
                                                                              │
[Tầng 4: Fine-tune Nộp]   Gộp Data Nền + VN + Copy-Paste (10%) ──► Fine-tune yolo26n-seg (640) ──► ONNX
```

### Tại Sao Chiến Thuật Này Tối Ưu Trong 1 Ngày?
1. **Không vẽ tay từ đầu:** Tận dụng sẵn polygon của **IDD** (rất giống VN: có `motorcycle`, `rider`, `autorickshaw`), **BDD100K** (ngày/đêm/mưa) và **Roboflow Universe** (`vietnam traffic`).
2. **Dùng Model lớn làm Teacher (`yolo26x-seg.pt` hoặc Model v1):** Tự động sinh polygon sắc nét cho ảnh Dashcam Việt Nam (`retina_masks=True`, `imgsz=960`), kết hợp rule hình học (`person` giao `motorcycle` → `Motorcyclist`) để giảm 80% công sửa nhãn.
3. **Khóa tập `val` đúng bối cảnh thi:** Toàn bộ tập `val` (80–120 ảnh) lấy từ **ảnh giao thông Việt Nam đã sửa tay sạch tuyệt đối**, giúp điểm `mAP@[.5:.95]` phản ánh đúng thực tế.
4. **Cải thiện theo hướng Data-Centric (Cell 21):** Vì `yolo26n-seg` và `imgsz=640` bị khóa cứng lúc chấm, thắng thua nằm ở **sửa nhầm lớp** (`Motorcycle`/`Motorcyclist`, `Bicycle`/`Cyclist`, `Truck`/`Other vehicle`), **xóa mask `< 64 px`**, và **Copy-Paste lớp hiếm**.

---

## 2. Tổ Chức GitHub & Hạ Tầng Online (Colab / Server GCP)

### 2.1. Cấu Trúc Repo GitHub
**Nguyên tắc:** GitHub chỉ chứa code và cấu hình; **tuyệt đối không push ảnh, file `.txt`, `.zip`, `.pt` lên Git.**

```text
allnaction-traffic-seg/
├── .gitignore                  # Chặn dataset/, data/, runs/, *.pt, *.onnx, *.zip, *.jpg, *.png
├── README.md
├── configs/
│   └── traffic_11_classes.yaml # Khớp chính xác 11 lớp ở Cell 10 của train_and_export.ipynb
├── scripts/
│   ├── prelabel_yolo_seg.py    # Prelabel ảnh VN + lọc mask < 64px + xuất data.yaml chuẩn CVAT
│   ├── remap_open_datasets.py  # Remap ID từ IDD / BDD100K / Roboflow về 0..10
│   ├── validate_dataset.py     # Check polygon >= 3 đỉnh, tọa độ [0,1], lọc trùng hash & leak val
│   └── copy_paste_aug.py       # Nhân bản mask lớp hiếm (< 100 instances)
└── notebooks/
    └── train_and_export.ipynb  # Notebook chuẩn cuộc thi (TEAM_ID cố định cho cả 5 người)
```

### 2.2. Nơi Lưu Trữ & Chạy GPU
- **Train GPU:** Dùng **Server GCP** của đội (kết nối qua VS Code Remote-SSH hoặc Colab *Connect to local runtime*) hoặc **Colab T4 / Kaggle GPU**.
- **Lưu Data & Weights:** Nén `.zip` để trên **Google Drive chung** (hoặc trực tiếp trên ổ SSD của Server GCP). Nếu dùng Colab, **bắt buộc giải nén vào `/content/`** trước khi train, không đọc ảnh trực tiếp từ Drive.

---

## 3. Phân Công 5 Thành Viên Chạy Song Song 100%

| Thành Viên | Vai Trò | Nhiệm Vụ Cụ Thể | Đầu Ra |
| :--- | :--- | :--- | :--- |
| **Người 1** | **Open Prelabel & Remap** | Tải IDD / BDD100K / Roboflow VN, chạy script remap chuẩn 11 lớp `0..10`, lọc bỏ nhãn lỗi. | `open_prelabel_v1.zip` |
| **Người 2** | **Thu Thập & Cắt Frame VN** | Kéo video Dashcam/Motovlog VN (ngày, đêm, mưa, giao lộ), cắt 1 frame mỗi 3–5 giây (tránh trùng lặp), đẩy qua `prelabel_yolo_seg.py`. | `vn_raw_prelabeled.zip` |
| **Người 3** | **CVAT Review & Khóa `Val`** | Sửa sạch biên mask trên CVAT (tách `Motorcyclist`/`Cyclist`, thêm `Other vehicle`, `Traffic sign`), chốt cứng tập `val` + lô `train` VN. | `val_locked.zip` + `vn_clean.zip` |
| **Người 4** | **Model Pretrain & Fine-tune** | Dựng Server/Colab GPU, khóa `TEAM_ID`, train **Model v1 (Pretrain trên Open Data)** → làm Teacher → Fine-tune `yolo26n-seg` (`imgsz=640`) & xuất ONNX. | `best.pt`, `submission.zip`, `run_report.json` |
| **Người 5** | **Data-Centric QC & Aug** | Viết script `validate_dataset.py` (chặn leak, trùng lặp, mask `< 64px`), soi **6 ảnh lỗi nhất (Bước 5)**, chạy `copy_paste` cứu lớp yếu. | `data_v2.zip` + Báo cáo lỗi từng lớp |

---

## 4. Lịch Trình Thực Chiến 1 Ngày (8 Giờ)

| Khung Giờ | Mốc Công Việc | Hành Động Cụ Thể Của Đội |
| :--- | :--- | :--- |
| **Giờ 0 – 1**<br>*(Setup & Prelabel)* | Chốt hạ tầng & Nguồn data | • Người 4 dựng GitHub + Server/Colab GPU, test `train_and_export.ipynb` (Bước 0).<br>• Người 1 tải & remap Open Data (`open_prelabel_v1.zip`).<br>• Người 2 cắt 250 frame Dashcam VN + chạy `prelabel_yolo_seg.py`.<br>• Người 5 chốt script `validate_dataset.py`. |
| **Giờ 1 – 3**<br>*(Model Pretrain + Chốt Val)* | Ra **Model v1** & **Val khóa cứng** | • Người 3 sửa sạch **100 ảnh VN** → đóng gói **`val_locked.zip`** (khóa cứng suốt giải).<br>• Người 4 train **Model v1 (Warm-up)** trên `open_prelabel_v1.zip` + `val_locked.zip` → lấy mốc điểm Baseline `mAP@[.5:.95]`. |
| **Giờ 3 – 5**<br>*(Pseudo-Label & Sửa VN)* | Nạp Data Việt Nam sạch | • Dùng **Model v1** (hoặc `yolo26x-seg.pt`) **Pseudo-Label** thêm 200–300 ảnh Dashcam VN mới.<br>• Người 1, 2, 3 cùng vào CVAT sửa nhanh lô ảnh VN này (tập trung lớp hay nhầm). |
| **Giờ 5 – 7**<br>*(Fine-tune v2 + Copy-Paste)* | Ép điểm **Mask mAP@[.5:.95]** | • Người 5 chạy `validate_dataset.py` gộp **Open + VN Clean + Copy-Paste** (đảm bảo mọi lớp `>= 100` instances, `imbalance <= 30x`).<br>• Người 4 fine-tune `yolo26n-seg` (`EPOCHS=80-100`, `imgsz=640`, `copy_paste=0.3`, `close_mosaic=10`, `mask_ratio=2`). |
| **Giờ 7 – 8**<br>*(Active Learning & Nộp)* | Sửa 6 ảnh lỗi nhất & Chốt | • Chạy Bước 5 trên `val_locked.zip`, soi **6 ảnh lỗi nhất** và bảng điểm từng lớp.<br>• Sửa lỗi nhãn/bổ sung đúng lớp yếu → Fine-tune nhanh vòng cuối (20–30 epoch) → Xuất `submission.zip` + `run_report.json`. |

---

## 5. Checklist Kiểm Soát Chất Lượng (Trước Khi Bấm Nộp)
- [ ] Mọi thành viên dùng chung 1 `TEAM_ID` (seed) duy nhất ở Cell 6.
- [ ] Bước 2 (`analyze_dataset`) báo `REPORT["ok"] == True`, không có rò rỉ `train`/`val`.
- [ ] Đã đối chiếu đúng 11 ID lớp (`0..10`) giữa CVAT, Open Data và Cell 10 của notebook.
- [ ] Đã chạy xong **Bước 4 (Xuất ONNX opset 17, `<= 42 MiB`)** và **Bước 5 (Chấm thử `onnxruntime` CPU)**.
- [ ] Đã tải đủ 2 file nộp bài: `submission.zip` và `run_report.json`.
