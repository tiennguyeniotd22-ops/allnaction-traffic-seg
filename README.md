# AllnAction · Traffic Instance Segmentation

Nhóm xây dựng bộ dữ liệu phân đoạn 11 lớp giao thông cho bài toán Robotaxi. Mô hình `yolo26n-seg` được đánh giá bằng Mask mAP@[.5:.95]. Mục tiêu trong ngày là 300–600 ảnh train và 80–120 ảnh validation đã khóa. [Playbook và phân công](ALLNACTION_1DAY_HACKATHON_PLAYBOOK.md)

## Nộp dữ liệu

1. Chọn đúng tiền tố của mình. Mỗi cặp ảnh và nhãn YOLO segmentation dùng cùng tên gốc, ví dụ `images/train/shardA_0001.jpg` và `labels/train/shardA_0001.txt`.
2. Kiểm tra các polygon sau khi xuất từ CVAT hoặc Roboflow. Đóng gói thư mục `images/train/` và `labels/train/` vào cùng một ZIP. Tập `val_locked.zip` do người phụ trách QC quản lý riêng.
3. Tải ZIP lên [thư mục AllnAction_Data](https://drive.google.com/drive/folders/1G06RBQwL3igPiVxMB3c-GYB_gwQvA8M0). Khi sửa dữ liệu, tăng số phiên bản để người train có thể tái lập lần chạy trước.
4. Mở [phiếu nộp dữ liệu](https://github.com/tiennguyeniotd22-ops/allnaction-traffic-seg/issues/new?template=nop-du-lieu.yml), dán link **của file ZIP**, ghi số ảnh, số file nhãn và nguồn ảnh rồi bấm **Create**. Giữ quyền truy cập Drive trong nhóm.
5. Người phụ trách QC kiểm tra khả năng tải file, cặp ảnh–nhãn, ID lớp `0..10`, tọa độ polygon và trùng ảnh với tập validation. Sau khi gộp, ghi phiên bản dataset vào Issue và đóng phiếu.

| Thành viên | Bối cảnh | Tên gói đầu tiên |
| --- | --- | --- |
| Người 1 · Shard A | Cao tốc, ban ngày | `shardA_v1.zip` |
| Người 2 · Shard B | Đô thị, giao lộ | `shardB_v1.zip` |
| Người 3 · Shard C | Đêm, mưa và tình huống khó | `shardC_v1.zip` |

## Làm việc với mã nguồn

- `main` chứa phiên bản dùng chung của nhóm.
- Người 4 làm pipeline huấn luyện trên `feat/train-pipeline`.
- Người 5 làm công cụ tạo nhãn và QC trên `feat/data-tools`.
- Ảnh, nhãn và ZIP nằm trong Drive `AllnAction_Data`. Weights và log huấn luyện dùng thư mục `AllnAction_Runs` theo playbook.
