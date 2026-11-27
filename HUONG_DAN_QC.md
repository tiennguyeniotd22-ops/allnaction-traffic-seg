# QC nhãn giao thông — tiêu chí bàn giao dữ liệu cho người train

**Mục tiêu của đội là nộp mô hình có thể vẽ đúng mask của từng đối tượng trong một ảnh đường phố mới.** Năm người sửa 60 ảnh/người để tạo một bộ 300 ảnh có đáp án tham chiếu. Mỗi đáp án cho biết đối tượng nào xuất hiện, thuộc lớp nào và đường bao của chính đối tượng đó nằm ở đâu.

## Khi mở một frame trong CVAT

| Cần kiểm tra | Cách quyết định |
|---|---|
| Có bỏ sót hoặc gán trùng không? | Quét toàn cảnh trước khi nhìn các polygon có sẵn. Thêm đối tượng bị bỏ sót, xóa nhãn trùng và tách hai phương tiện đi sát nhau thành hai instance. |
| Lớp và đơn vị đối tượng đã đúng chưa? | Người đang lái xe máy và xe là **một** `Motorcyclist`. Xe máy trống là `Motorcycle`. Áp dụng cùng quy ước cho `Cyclist` và `Bicycle`. Rà cả các lớp ngoài phần trọng tâm của mình. |
| Đường bao có bám đúng vật thể không? | Phóng to vật nhỏ, vùng che khuất, ảnh mưa và đêm. Chỉnh các cạnh ăn sang nền hoặc sang vật thể bên cạnh. Với biển báo, kiểm tra vùng mặt biển. |
| Còn trường hợp chưa thống nhất? | Ghi Job, số frame, đối tượng và điểm phân vân vào issue. Chụp vùng ảnh nếu cần trao đổi với người phụ trách lớp. |

## Người thứ hai duyệt gì?

1. Chủ Job rà đủ 60 ảnh, lưu thay đổi và gửi Đăng các frame còn vướng. Người duyệt khác xem các frame này và chọn thêm một số ảnh từ mỗi nhóm cảnh để đối chiếu theo bảng trên.
2. Nếu phát hiện lỗi lặp, chủ Job rà lại những ảnh có cùng tình huống. Ví dụ, khi một người lái và xe bị tách thành `Pedestrian` cộng `Motorcycle`, kiểm tra lại các frame có xe máy trước khi bàn giao.
3. Với mỗi lỗi, ghi ngắn gọn **Job/frame · lỗi gì · cần sửa gì · ai đã kiểm tra lại**. Frame đạt khi đối tượng, lớp và đường biên đã được kiểm tra, các lỗi phát hiện đã sửa và lưu. Khi đủ 60 ảnh và đã xử lý các issue, chủ Job chuyển sang `Completed`.

## Từ Job hoàn thành đến bài nộp

| Bàn giao | Đầu ra kiểm tra được |
|---|---|
| **Đăng xuất từ CVAT sau duyệt, Trinh kiểm tra bản xuất** | Bộ ảnh và file nhãn YOLO-seg cùng tên gốc. Mỗi dòng nhãn có một ID lớp từ 0 đến 10 và tọa độ polygon chuẩn hóa, ít nhất ba đỉnh khác nhau. Mở thử một frame đã sửa để xác nhận bản xuất có đúng thay đổi. [Bảng lớp chuẩn](configs/traffic_11_classes.yaml) |
| **Đăng khóa danh sách train/val, Sang chạy notebook** | Ảnh cùng video/cảnh và các bản tăng cường nằm trong cùng một split. Mô hình học từ ảnh và nhãn của tập train. Sang dùng tập val đã khóa để đo mask mAP và so sánh các vòng. [Notebook của đội](notebooks/train_and_export.ipynb) |
| **Sang xuất mô hình, Tiến nộp bài** | `submission.zip` tối đa **42 MB**, chứa đúng **một file `model.onnx`** ở gốc ZIP. Đây là mô hình `yolo26n-seg`, FP32, đầu vào 640, đã qua bước chấm thử của notebook. |
| **Cả nhóm bàn giao báo cáo** | Slide trình bày nguồn, giấy phép, ví dụ trước/sau QC và kết quả các vòng. Giữ notebook, dữ liệu đúng phiên bản và `run_report.json` để tái lập. [Quy trình và nhiệm vụ từng người](QUY_TRINH_VA_PHAN_CONG.md) |
