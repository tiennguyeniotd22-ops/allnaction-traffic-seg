# Đề xuất tăng điểm mask · Đội 21

Trưởng nhóm đã duyệt A và C, sau đó cho phép bổ sung dữ liệu theo B. A đã rà 50 cảnh. B đã chọn 31 cảnh COCO và đối chiếu với tập train, val, holdout và tập chẩn đoán. Bảng dưới ghi kết quả mốc `team21-qc20`. [Guideline BTC](labeling_guideline.md) · [Quy trình QC của đội](HUONG_DAN_QC.md) · [Nhật ký thí nghiệm](EXPERIMENTS.md)

## Mốc so sánh

ZIP của pilot đã được chấm lại trên CPU, dùng đúng notebook của đội, seed 21 và 140 ảnh validation cố định. Bộ chấm nhận 1.588 instance hợp lệ. **Mask mAP@[.50:.95] đạt 24,13/100, AP50 đạt 41,55 và AP75 đạt 24,49.** Tập này có 10 lớp có ground truth hợp lệ. Điểm public/private của ZIP cần kết quả từ hệ thống cuộc thi. [Replay và SHA của ZIP](experiments/team21-qc20/replay.json)

| Quan sát trên val140 | Câu hỏi cần kiểm trên train |
|---|---|
| Motorcycle AP 7,45. Với confidence ≥ 0,25 và mask IoU ≥ 0,50, chỉ 9/54 xe máy trống được khớp đúng theo phép ghép một-một. | Nhãn train có bỏ xe đỗ, dãy xe sát nhau hoặc gộp người đứng cạnh vào xe không? |
| Cyclist AP 15,66. Khi xét từng GT riêng, 12/33 người đi xe đạp có mask dự đoán lớp Pedestrian phủ trúng ở IoU ≥ 0,50 và confidence ≥ 0,25. | Mask người–xe có bị tách khi chuyển từ nhãn nguồn sang 11 lớp hoặc khi xuất từ CVAT không? |
| Truck AP 19,91. 14/82 GT Truck có mask dự đoán lớp Car phủ trúng theo cùng phép chẩn đoán. | Kiểm lại bán tải, xe tải nhẹ, đầu kéo, rơ-moóc và cách ánh xạ lớp nguồn. |
| Traffic light AP 15,58. Ở confidence ≥ 0,25, số khớp đúng là 62/154 ở IoU 0,50 và 24/154 ở IoU 0,75. | Nhãn có ôm đủ vỏ hộp, bỏ cột, tách các hộp cùng cột và xử lý đúng mốc kích thước hay chưa? |

Các quan sát xác định nhóm ảnh nên kiểm. Muốn kết luận lỗi nhãn phải có ảnh gốc và mask đối chiếu. Phép đếm chồng mask theo từng GT có thể dùng lại cùng một dự đoán cho nhiều GT. Ngưỡng confidence 0,25 chỉ dùng để phân tích lỗi. [Số đếm, ngưỡng và ma trận nhầm lớp](experiments/team21-qc20/error-profile.json)

## Phạm vi xin duyệt

| Gói | Thao tác đề xuất | Đầu ra để nghiệm thu |
|---|---|---|
| **A · Sửa nhãn train hiện có** | Bắt đầu bằng 40 cảnh train khác nhau, ưu tiên 20 cảnh xe hai bánh, 10 cảnh xe tải/đầu kéo và 10 cảnh biển/đèn. Phân mỗi cảnh vào một nhóm chính và vẫn rà đủ 11 lớp. Thêm 10 cảnh train ngẫu nhiên theo seed 21 để phát hiện lỗi ngoài nhóm ưu tiên. Kiểm ảnh trước overlay, áp dụng mục 2–4 của BTC. | Danh sách lấy mẫu, số lỗi trên từng nhóm, ảnh và ID bị sửa, căn cứ BTC, trước/sau, kiểm lại bản xuất. Nếu thiếu cảnh phù hợp thì ghi số thực tế. Nếu không xác nhận được lỗi thì kết thúc A tại báo cáo. |
| **B · Bổ sung cảnh còn thiếu** | Sau A, nếu số cảnh đa dạng vẫn ít, chọn tối đa 60 cảnh train mới. Ưu tiên xe máy đỗ/dắt, xe đạp có người, ba gác/xích lô, biển và đèn nhìn nghiêng hoặc từ sau. Tìm cảnh Việt Nam trước, kiểm nguồn và quyền sử dụng, loại trùng ảnh và cảnh với train/val, holdout lớp 3 và tập Mapillary chẩn đoán. | Manifest ghi nguồn, giấy phép, ID gốc, số cảnh nhận/loại, số instance từng lớp. Chỉ nhận cảnh đã rà toàn khung. |
| **C · Đo hiệu quả** | Chấm lượt full đang chạy để có mốc riêng. Với A, dùng bộ pilot làm gốc, chỉ thay nhãn đã xác nhận. Giữ pretrained weights, notebook, seed 21, 20 epoch, batch 8, optimizer, augmentation và split như pilot. Nếu B được duyệt, so A+B với A theo cùng công thức. Mỗi lần chỉ chạy một fit GPU. | Chấm từ chính ZIP, ghi chênh mask mAP, AP50/AP75, AP từng lớp, tập Việt Nam và khoảng bootstrap ghép cặp theo cảnh. So `best` và `last` cùng cách cho các lượt. |

Đối chứng A so nhãn đã sửa với pilot. Đối chứng A+B thêm 31 ảnh vào đúng bộ A. [Sổ quyết định A](experiments/overnight/team21-btc-decisions.json) · [Manifest B](experiments/overnight/team21-road-manifest.json)

## Quy tắc chọn bài

Giữ `yolo26n-seg`, 640 × 640, FP32, opset 17, TEAM_ID và seed 21. Chỉ chọn ứng viên có ZIP hợp lệ, parity PyTorch–ONNX và kết quả chấm lại trên val140 từ ZIP. Xếp theo mask mAP@[.50:.95] trên cùng val. Báo kèm mức thay đổi trên cảnh Việt Nam và các lớp yếu. Nếu khoảng ước lượng chênh lệch còn chứa 0 thì ghi rõ kết quả chưa cho thấy lợi ích ổn định. Giữ bản đã xác minh cao nhất để trưởng nhóm quyết định nộp.

Lớp 3 có 0 GT hợp lệ trên val140, nên ghi “thiếu GT”. Báo AP của lớp này trên holdout riêng có 7 GT, 6 ảnh và 3 nhóm cảnh. Quy mô đó chỉ đủ cho chẩn đoán ban đầu. Vòng đề xuất dùng lỗi val để đặt giả thuyết và kiểm trên train. [Mốc val140](experiments/team21-qc20/replay.json)
