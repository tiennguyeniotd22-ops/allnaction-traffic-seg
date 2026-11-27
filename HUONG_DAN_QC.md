# Hướng dẫn review polygon trên CVAT · 11 lớp giao thông

**Mỗi người rà đủ 60 ảnh trong Job của mình. Trên từng ảnh, kiểm tra mọi đối tượng nhìn thấy và nhận diện được thuộc 11 lớp.** Trọng tâm cá nhân dùng cho lượt kiểm tra cuối. Ví dụ, trong ảnh có một người đang đạp xe, hai ô tô và một biển báo, Trinh cần kiểm tra bốn instance gồm một `Cyclist`, hai `Car` và một `Sign`. [Mở Job theo phân công](QUY_TRINH_VA_PHAN_CONG.md)

## Tra lớp khi sửa một đối tượng

| ID | Nhãn trên CVAT | Cách nhận diện và phần cần vẽ |
|---:|---|---|
| 0 | `Car` | Ô tô con gồm sedan, SUV, minivan, xe 7 chỗ, **xe bán tải (pickup)**, taxi, xe điện du lịch và xe cứu thương dạng van. Xe chở khách 9–15 chỗ thuộc lớp này. Vẽ thân từng xe. |
| 1 | `Bus` | Xe buýt, xe khách từ 16 chỗ trở lên. Nếu góc ảnh khiến xe khách nhỏ khó phân biệt với minivan, đánh dấu để duyệt chung. |
| 2 | `Truck` | Xe tải các cỡ, xe ben, xe bồn, xe cứu hỏa. Với đầu kéo, vẽ cabin và thân đầu kéo. Nếu thấy rơ-moóc phía sau, vẽ riêng phần đó theo lớp 3. |
| 3 | `Other vehicle` | Ba gác, xích lô, xe lam, xe công trình, máy cày, rơ-moóc hoặc sơ-mi rơ-moóc. Dùng khi nhận ra loại phương tiện. Nếu chỉ thấy một mảng mờ chưa nhận diện được, mở issue. |
| 4 | `Motorcycle` | Xe máy, xe tay ga, xe điện hai bánh kiểu xe máy khi yên trống. Áp dụng cho cả xe đang đỗ và xe có người **đi bộ dắt**. |
| 5 | `Motorcyclist` | Có ít nhất một người ngồi trên xe máy. Vẽ **một instance cho xe và tất cả người ngồi trên xe**, kể cả người ngồi sau hoặc khi xe dừng đèn đỏ. |
| 6 | `Bicycle` | Xe đạp hoặc xe đạp điện kiểu xe đạp khi yên trống, kể cả xe đang được dắt. |
| 7 | `Cyclist` | Có người ngồi trên xe đạp. Vẽ một instance bao phần nhìn thấy của xe và người ngồi trên xe. |
| 8 | `Pedestrian` | Người đi bộ, đứng, ngồi ngoài phương tiện, trẻ em hoặc người đi bộ dắt xe. Mỗi người là một instance. |
| 9 | `Sign` | Biển báo giao thông, biển chỉ dẫn, biển công trường. Vẽ theo mép **mặt biển**, chừa cột đỡ. Hai mặt biển phân biệt được trên cùng cột có hai polygon. |
| 10 | `Traffic light` | Vẽ theo đường viền **cả hộp đèn tín hiệu**. Ví dụ, đèn đỏ đang sáng thì polygon vẫn bao cả cụm chứa các bóng. Chừa cột và tay đỡ. |

Tên lớp và ID dùng khi xuất YOLO-seg nằm trong [cấu hình 11 lớp](configs/traffic_11_classes.yaml).

## Khi mở một frame trong CVAT

| Cần kiểm tra | Cách quyết định |
|---|---|
| Có bỏ sót hoặc gán trùng không? | Quét toàn cảnh trước khi nhìn các polygon có sẵn. Thêm đối tượng bị bỏ sót, xóa nhãn trùng và tách hai phương tiện đi sát nhau thành hai instance. |
| Lớp và đơn vị đối tượng đã đúng chưa? | Người đang lái xe máy và xe là **một** `Motorcyclist`. Xe máy trống là `Motorcycle`. Áp dụng cùng quy ước cho `Cyclist` và `Bicycle`. Rà cả các lớp ngoài phần trọng tâm của mình. |
| Đường bao có bám đúng vật thể không? | Phóng to vật nhỏ, vùng che khuất, ảnh mưa và đêm. Chỉnh các cạnh ăn sang nền hoặc sang vật thể bên cạnh. Với biển báo, kiểm tra vùng mặt biển. |
| Còn trường hợp chưa thống nhất? | Ghi Job, số frame, đối tượng và điểm phân vân vào issue. Chụp vùng ảnh nếu cần trao đổi với người phụ trách lớp. |

**Cách rà trên CVAT.** Tạm ẩn polygon, nhìn ảnh gốc từ trái sang phải và từ gần ra xa. Bật lại polygon để so từng đối tượng. Phóng to vùng cần sửa rồi thu nhỏ để kiểm lại toàn cảnh. Với vật bị khuất hoặc bị cắt ở mép ảnh, bám phần còn nhìn thấy và dừng ở mép ảnh. Với vùng tối hoặc mưa che mất chi tiết, ghi rõ chỗ chưa phân biệt được để người thứ hai xem. Kiểm tra đường bao khép kín, có ít nhất ba đỉnh khác nhau và các cạnh không tự cắt.

| Tình huống dễ gán sai | Kết quả mong đợi sau khi sửa |
|---|---|
| Một người đi bộ dắt xe đạp, bên cạnh có một người đang ngồi trên xe máy | Ba instance gồm `Pedestrian`, `Bicycle` và `Motorcyclist`. Xóa polygon người bị gán trùng trên xe máy. |
| Hai xe máy áp sát nhau, mỗi xe có người ngồi | Hai `Motorcyclist`. Kiểm tra vùng giữa hai xe để đường bao của xe này không ăn sang xe kia. |
| Ảnh chụp để nhận diện biển báo, nhãn nháp chỉ khoanh biển | Quét cả mặt đường và vỉa hè. Thêm người, xe, đèn còn thiếu nếu nhìn rõ. |

**Hai ảnh thật để thống nhất cách duyệt.** Tìm theo tên ảnh trong Job. Các số bên dưới là số nhãn **nháp** để đối chiếu với ảnh gốc. Khi nháp sai, sửa theo ảnh.

- **Job 2 của Trân** · `vn_sign_54a38a490fbd063bd4.jpg`. Bản nháp có 5 `Motorcycle` và 4 `Motorcyclist`. Kiểm từng yên xe, người ngồi sau và đường biên giữa các xe gần nhau.
- **Job 5 của Trinh** · `vn_sign_692b88cfef67f51d57.jpg`. Bản nháp có 1 `Bicycle`, 2 `Pedestrian`, 2 `Motorcycle` và 2 `Sign`. Xem lại quan hệ người–xe, sau đó kiểm cả các mặt biển và phần còn lại của ảnh.

## Khi nào một frame đạt?

- [ ] Đã xem ảnh gốc và rà toàn khung. Mỗi đối tượng nhận diện được thuộc 11 lớp đều có nhãn.
- [ ] Đúng lớp, đúng số instance. Đã xử lý nhãn thiếu, nhãn trùng và người–xe bị tách hoặc gộp sai.
- [ ] Polygon bám phần nhìn thấy. Đã kiểm vùng che khuất, mép ảnh, nền lọt vào mask, cột biển và hộp đèn nếu có.
- [ ] Đã lưu trên CVAT. Nếu frame còn điểm chưa thống nhất, ghi issue và giữ frame đó chờ duyệt.

## Người thứ hai duyệt gì?

1. Chủ Job rà đủ 60 ảnh, lưu thay đổi và gửi Đăng các frame còn vướng. Người duyệt thứ hai kiểm tất cả frame có issue, rồi lấy mẫu từ cả bốn nhóm ảnh trong Job gồm ban ngày, ban đêm, trời mưa và giao thông Việt Nam.
2. Nếu phát hiện lỗi lặp, chủ Job rà lại những ảnh có cùng tình huống. Ví dụ, khi một người lái và xe bị tách thành `Pedestrian` cộng `Motorcycle`, kiểm tra lại các frame có xe máy trước khi bàn giao.
3. Với mỗi lỗi, ghi ngắn gọn **Job/frame · lỗi gì · cần sửa gì · ai đã kiểm tra lại**. Frame đạt khi đối tượng, lớp và đường biên đã được kiểm tra, các lỗi phát hiện đã sửa và lưu. Khi đủ 60 ảnh và đã xử lý các issue, chủ Job chuyển sang `Completed`.

Ví dụ **cách ghi issue** (tình huống minh họa) · `Job 5 / frame … / người đang đi bộ dắt xe / nháp đang gộp thành Cyclist / sửa thành Pedestrian + Bicycle / chờ người duyệt mở lại frame`. Khi bàn giao, chủ Job ghi số frame đã rà trên 60, các frame đã sửa lại và tên người duyệt chéo. Nếu quyền CVAT chỉ cho phép mở Job của mình, hai người cùng xem trên màn hình của chủ Job.

## Từ Job hoàn thành đến bài nộp

| Bàn giao | Đầu ra kiểm tra được |
|---|---|
| **Đăng xuất từ CVAT sau duyệt, Trinh kiểm tra bản xuất** | Bộ ảnh và file nhãn YOLO-seg cùng tên gốc. Mỗi dòng nhãn có một ID lớp từ 0 đến 10 và tọa độ polygon chuẩn hóa, ít nhất ba đỉnh khác nhau. Mở thử một frame đã sửa để xác nhận bản xuất có đúng thay đổi. [Bảng lớp chuẩn](configs/traffic_11_classes.yaml) |
| **Đăng đối chiếu danh sách train/val, Sang chạy notebook** | Với vòng so sánh trước/sau review, ghép nhãn đã sửa về đúng tên ảnh theo danh sách 240 train và 60 val đã khóa ở lần 1. Dùng cùng 60 ảnh và cùng nhãn val đã duyệt để chấm cả hai mô hình. [Notebook của đội](notebooks/train_and_export.ipynb) |
| **Sang xuất mô hình, Tiến nộp bài** | `submission.zip` tối đa **42 MB**, chứa đúng **một file `model.onnx`** ở gốc ZIP. Đây là mô hình `yolo26n-seg`, FP32, đầu vào 640, đã qua bước chấm thử của notebook. |
| **Cả nhóm bàn giao báo cáo** | Slide trình bày nguồn, giấy phép, ví dụ trước/sau QC và kết quả các vòng. Giữ notebook, dữ liệu đúng phiên bản và `run_report.json` để tái lập. [Quy trình và nhiệm vụ từng người](QUY_TRINH_VA_PHAN_CONG.md) |
