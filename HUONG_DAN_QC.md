# QC polygon giao thông trên CVAT · Đội 21

Dùng **guideline của ban tổ chức** để quyết định đối tượng nào được gán, lớp nào đúng và pixel nào thuộc mask. Tài liệu này thay bản QC cũ và hướng dẫn cách kiểm một frame, ghi quyết định và nghiệm thu bản xuất của đội. Mỗi thành viên rà **đủ 11 lớp** trên ảnh mình nhận. Trọng tâm cá nhân chỉ quyết định vùng cần xem kỹ ở lượt cuối. [Guideline BTC, mục 1–6](labeling_guideline.md) · [Phân công](QUY_TRINH_VA_PHAN_CONG.md)

## Duyệt một frame

1. **Ẩn nhãn.** Xem ảnh gốc toàn khung, từ gần ra xa. Đếm riêng xe, người, mặt biển và hộp đèn. Kiểm cả vật bị cắt ở mép, bị che và mặt sau biển/đèn. Với cảnh đêm, chỉ gán xe khi nhận ra hình thể. Bỏ ảnh trên áp phích, phản chiếu, vật phía sau kính showroom và người trong ô tô.
2. **Đo phần nhìn thấy khi sát ngưỡng.** Trên ảnh gốc kích thước `W × H`, so cạnh ngắn của hộp bao phần nhìn thấy với khoảng `0,0125 × max(W,H)`. Ví dụ ảnh 1280 × 720 có mốc khoảng 16 px. Dưới mốc thì bỏ. Sát phía trên mà phải đoán thì cũng bỏ. Khi vật đủ lớn và nhận diện được nhưng khó chọn lớp, dùng cây quyết định của BTC rồi ghi cách xử lý để cả đội dùng lại. Cảnh báo diện tích mask dưới 64 px² ở thang 640 của pipeline là phép kiểm kỹ thuật riêng. Không dùng cảnh báo này để xóa nhãn hàng loạt. [Mục 2 và 5 của BTC](labeling_guideline.md)
3. **Bật nhãn và so từng instance.** Phóng to chỗ tiếp giáp. Vật trước giữ pixel, vật sau nhận phần còn nhìn thấy. Loại nền, bóng đổ, phản chiếu và quầng đèn. Giữ các phần bị ngăn bởi vật mảnh trong cùng một instance. Thu nhỏ để tìm đối tượng bị bỏ sót sau khi sửa.
4. **Ghi quyết định và lưu.** Với mỗi ca còn tranh luận, ghi Job, frame, tên ảnh, ID instance, quy tắc BTC và điểm cần người thứ hai xác nhận. Người duyệt mở ảnh gốc trước khi xem phương án của người gán.

## Các ca cần đối chiếu khi bật overlay

| Dấu hiệu trên ảnh | Quyết định cần nhìn thấy trong mask |
|---|---|
| Người ngồi trên xe hai bánh, có người ngồi sau hoặc hàng buộc chắc | Một `Motorcyclist` (5) hoặc `Cyclist` (7) gồm xe và người. Giữ cả hành khách, hàng gắn chắc nhìn thấy và áo mưa trùm người–xe. Mỗi xe trong dòng đông có instance riêng. |
| Xe đỗ, người dắt xe hoặc đang bước xuống | Kiểm vị trí trên yên. Còn ngồi thì vẫn là lớp có người. Người đi bộ dắt xe thành `Pedestrian` (8) và xe thành `Motorcycle` (4) hoặc `Bicycle` (6). Xe điện kiểu tay ga và xe đạp điện kiểu có bàn đạp theo hai nhánh tương ứng. |
| Xe lớn hoặc dễ nhầm lớp | Bán tải và xe du lịch 9–15 chỗ thuộc `Car` (0). Xe buýt/khách từ 16 chỗ là `Bus` (1). Đầu kéo và thân đầu kéo là `Truck` (2), rơ-moóc là một `Other vehicle` (3) riêng. Xe ben thuộc `Truck`, máy xúc và xe cẩu thuộc `Other vehicle`. |
| Xích lô, ba gác, xe kéo tay | Xích lô và ba gác gộp người ngồi/đạp cùng xe vào lớp 3. Xe kéo tay được bỏ, người kéo là lớp 8. |
| Người bế trẻ, cầm ô hoặc dùng xe lăn | Người bế/cõng trẻ là một instance. Mask người gồm ô đang cầm, nón và túi đeo. Người ngồi xe lăn và người đẩy phía sau là hai người riêng, bỏ xe lăn trong trường hợp này. |
| Biển hoặc đèn chung cột | Mỗi mặt biển, kể cả biển phụ và lưng biển, là một `Sign` (9). Mỗi hộp đèn, kể cả hộp dành cho người đi bộ và mặt sau, là một `Traffic light` (10). Vẽ cả vỏ hộp và bóng trong hộp. Chừa cột, giá đỡ và nền. Biển tên đường chỉ gán nếu là biển giao thông chính thức. |

Các tình huống còn lại tra trực tiếp theo từng lớp và bảng ca đặc biệt. [Mục 3–4 của BTC](labeling_guideline.md)

## Duyệt chéo và đóng lỗi

- Chủ Job rà toàn bộ frame. Người duyệt thứ hai kiểm hết các ca có issue và lấy thêm mẫu ngẫu nhiên từ các cảnh có trong Job. Ghi seed, danh sách ảnh, nguồn và điều kiện sáng. Báo riêng lỗi phát hiện trong mẫu ngẫu nhiên và lỗi phát hiện trong nhóm chủ động chọn.
- Ví dụ ghi một lỗi: `Job 5 / frame 18 / ảnh … / lớp 7 → 6 + 8 / người đi bộ dắt xe / BTC §3 Bicycle / người sửa … / người duyệt … / trạng thái …`. Đây là ví dụ cách ghi. Nếu chỉ một người kiểm, ghi “tự kiểm” và để trống tên người duyệt chéo.
- Khi thấy lỗi lặp, tìm lại các ảnh train cùng tình huống hoặc cùng nguồn pre-label. Ghi số frame đã rà, số lỗi thiếu/thừa, sai lớp, gộp/tách sai và sai đường biên. Frame đạt khi đã so toàn khung, sửa và lưu các lỗi xác nhận. Ca còn tranh luận được giữ lại để phân xử trước khi đưa vào bản train.

## Nghiệm thu bản xuất

| Kiểm tra | Điều kiện đưa bản xuất vào train |
|---|---|
| Đúng ảnh, lớp và tọa độ | Ghép ảnh với nhãn cùng tên. ID 0–10 đúng thứ tự chính thức. Mỗi polygon có ít nhất ba đỉnh phân biệt, tọa độ hữu hạn trong [0,1], diện tích dương, đường bao hợp lệ. Đối chiếu ảnh có nhãn trống. [Cấu hình lớp](configs/traffic_11_classes.yaml) |
| Một đối tượng có nhiều mảnh | Giữ ID nhóm trong CVAT. Mở lại mask từ file YOLO-seg bằng bộ giải mã của notebook và so với mask gốc ở cùng kích thước. Đếm lại instance, xem phần bị che, lỗ và vùng mảnh. Nếu xuất thành hai dòng độc lập cho một vật hoặc cầu nối ăn vào vật trước, treo ca đó và sửa cách biểu diễn trước khi train. |
| Biên và vùng chồng | Overlay sau xuất phải giữ phần nhìn thấy và thứ tự che khuất. Kiểm mọi frame có sửa nhóm/mask, cùng một mẫu frame bình thường. Ghi ảnh, ID và vị trí pixel khi mask sau xuất khác quyết định đã duyệt. |
| Split và truy vết | Bản dùng để so model hiện tại khóa 140 ảnh val. Đối chiếu SHA ảnh và nhãn, kiểm cả cảnh gần trùng giữa train/val. Lưu phiên bản dữ liệu, log sửa và nguồn/giấy phép. Mọi sửa nhãn đề xuất ở vòng này thuộc train. |
| Bàn giao kết quả | Lưu `run_report.json`, checkpoint, seed 21, SHA dữ liệu và ZIP. Chấm lại từ ZIP theo đúng notebook trên cùng val140. Ghi mask mAP@[.50:.95], AP50, AP75 và AP theo lớp. [Notebook của đội](notebooks/train_and_export.ipynb) |

[Đề xuất thử nghiệm đang chờ trưởng nhóm duyệt](DE_XUAT_TANG_DIEM_MASK.md)
