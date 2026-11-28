# Hướng dẫn gán nhãn Instance Segmentation (bản dành cho học viên)

Đây là quy ước 11 lớp mà bộ chấm dùng. Hãy gán (hoặc ánh xạ nhãn của public dataset) theo cùng quy ước để mô hình học đúng điều mà bộ chấm kỳ vọng. Điểm của đội = 100 x mask mAP@[.5:.95], nên mask càng khớp với vật thể thì điểm càng cao.

## 1. Danh sách 11 lớp

Thứ tự và id dưới đây là cố định, dùng trong nhãn, trong `data.yaml` và trong đầu ra của mô hình.

| ID | Lớp | Tiếng Việt | Mô tả ngắn |
|---|---|---|---|
| 0 | `Car` | Ô tô con | Sedan, SUV, minivan, xe 7 chỗ, bán tải, taxi, xe điện du lịch (golf cart), xe cứu thương dạng van |
| 1 | `Bus` | Xe buýt / xe khách | Xe buýt, xe khách từ 16 chỗ trở lên |
| 2 | `Truck` | Xe tải | Xe tải các cỡ, xe ben, xe bồn, xe cứu hỏa, **đầu kéo** (phần cabin) |
| 3 | `Other vehicle` | Phương tiện khác | Xe ba gác máy, xích lô, xe lam, xe cẩu, xe công trình, **rơ-moóc / sơ-mi rơ-moóc**, xe máy cày |
| 4 | `Motorcycle` | Xe máy **không có người ngồi** | Xe máy, xe tay ga, xe điện 2 bánh kiểu xe máy, đang đỗ hoặc đang dắt |
| 5 | `Motorcyclist` | Xe máy **có người ngồi** | Một instance gồm xe và toàn bộ người trên xe |
| 6 | `Bicycle` | Xe đạp **không có người** | Xe đạp, xe đạp điện kiểu xe đạp, đang đỗ hoặc đang dắt |
| 7 | `Cyclist` | Xe đạp **có người đi** | Một instance gồm xe đạp và người đi |
| 8 | `Pedestrian` | Người đi bộ | Mọi người không ngồi trên phương tiện |
| 9 | `Sign` | Biển báo giao thông | Biển báo hiệu đường bộ, biển chỉ dẫn, biển tạm công trường |
| 10 | `Traffic light` | Đèn tín hiệu giao thông | Hộp đèn (đèn xe, đèn người đi bộ) |

Lưu ý: Motorcyclist là "xe máy có người ngồi", Cyclist là "xe đạp có người ngồi".

## 2. Nguyên tắc chung

1. **Mỗi đối tượng là một instance**, một mask riêng, đúng một lớp. Mask chỉ phủ phần **nhìn thấy** của đối tượng.
2. **Mask không chồng lên nhau.** Khi hai đối tượng che nhau, mask của vật phía trước giữ pixel, mask của vật phía sau chỉ phủ phần còn lại nhìn thấy.
3. **Gán tất cả đối tượng đủ lớn**, không chỉ đối tượng chính. Nếu cạnh ngắn của hộp bao nhìn thấy nhỏ hơn khoảng **1,25% cạnh dài của ảnh** (khoảng 16 px với ảnh cạnh dài 1280 px) thì không gán. Giữa ngưỡng đó và kích thước lớn hơn một chút, nếu phải đoán thì bỏ qua.
4. **Đối tượng bị cắt bởi mép ảnh** vẫn gán, mask chỉ phủ phần nằm trong ảnh.
5. **Biên mask bám sát đối tượng:** không để hở nền lớn, không tràn sang vật khác. Sai số vài pixel ở biên là chấp nhận được. Mask không phủ bóng đổ, ánh phản chiếu, ánh đèn pha chiếu ra; chỉ phủ vật thể.
6. Đối tượng bị che bởi vật mảnh (cột điện, dây, lan can, cành cây) vẫn là **một instance**. Trong công cụ gán nhãn dùng nhiều polygon cho cùng một instance, hoặc nhóm các mảnh lại.
7. **Không gán** vật thể thật ở phía sau kính (trong showroom, cửa hàng), người ngồi trong ô tô nhìn thấy qua cửa kính, vật trong ảnh áp phích hoặc màn hình quảng cáo, mô hình, ma-nơ-canh, tượng.
8. Khi **không chắc** về lớp, áp dụng cây quyết định ở mục 5. Nếu vẫn không quyết được, chọn lớp hợp lý nhất và ghi lại quyết định đó trong tài liệu quy ước của đội để cả đội gán nhất quán. Không bỏ trống đối tượng chỉ vì khó.

## 3. Quy tắc từng lớp

### Car (0)
- Gồm: sedan, hatchback, SUV, minivan, xe 7 chỗ, xe bán tải, taxi, xe điện du lịch, xe cứu thương dạng van, van nhỏ chở khách.
- Mask phủ toàn bộ thân xe nhìn thấy, gồm gương và bánh xe. Không phủ bóng dưới gầm, không phủ phản chiếu trên kính ngoài phần kính của chính xe.
- Xe có cửa kính nhìn thấy người bên trong: người bên trong **không** gán thêm.

### Bus (1)
- Gồm: xe buýt thành phố, xe buýt điện, xe khách và xe limousine từ 16 chỗ trở lên.
- Xe 9-15 chỗ kiểu "xe du lịch" thuộc **Car**.
- Quảng cáo dán bên hông xe vẫn là một phần của xe.

### Truck (2)
- Gồm: xe tải thùng, xe tải nhẹ, xe ben, xe bồn, xe trộn bê tông, xe cứu hỏa, xe chở rác, xe đầu kéo (phần cabin và thân đầu kéo).
- **Xe đầu kéo kéo rơ-moóc:** phần đầu kéo là `Truck`, phần rơ-moóc hoặc sơ-mi rơ-moóc là `Other vehicle`.
- Hàng hóa trên thùng xe nằm trong mask của xe. Hàng cao vượt hẳn thùng vẫn phủ cùng xe.

### Other vehicle (3)
- Gồm: xe ba gác máy, xe tuk-tuk, xe lam, xích lô, xe lôi, xe cẩu, xe công trình (máy xúc, xe lu, xe nâng), xe máy cày, xe rác tự chế, rơ-moóc và sơ-mi rơ-moóc, xe kéo, và **bất kỳ phương tiện giao thông nào không thuộc lớp khác**.
- Xe đẩy tay, xe kéo tay không phải phương tiện cơ giới: **không gán** (người kéo vẫn là Pedestrian).
- Xe điện 3 bánh nhỏ của người khuyết tật: `Other vehicle`.

### Motorcycle (4) và Motorcyclist (5)
- `Motorcycle`: xe máy, xe tay ga, xe mô tô, xe điện 2 bánh kiểu xe máy, **không có ai ngồi trên xe**. Gồm xe đang đỗ và xe đang được dắt (người đứng cạnh).
- `Motorcyclist`: xe máy **có ít nhất một người ngồi trên xe**. **Một instance duy nhất** phủ xe và toàn bộ người ngồi trên xe, gồm người chở phía sau, trẻ em ngồi trước, hàng hóa buộc vào xe.
- Người đã ngồi lên xe, dù xe đang dừng đèn đỏ, đang chờ hay đang nghe điện thoại: **vẫn là Motorcyclist**. Người ngồi trên yên xe đang đỗ cũng vậy.
- Người đứng cạnh xe, đang dắt hoặc đẩy xe: **Pedestrian** cho người, **Motorcycle** cho xe (hai instance riêng).
- Người mặc áo mưa trùm cả người và xe: vẫn một instance Motorcyclist phủ cả áo mưa.
- Dãy xe máy đông chạy sát nhau: **mỗi xe một instance**. Nếu hai xe chồng lấn nặng, mask phía trước giữ pixel, phía sau phần còn lại.

### Bicycle (6) và Cyclist (7)
- `Bicycle`: xe đạp, xe đạp điện kiểu xe đạp (có bàn đạp, khung xe đạp), **không có người**, kể cả đang đỗ hoặc đang dắt.
- `Cyclist`: xe đạp **có người đi**. **Một instance** phủ xe và người.
- **Xe đạp điện dạng xe máy điện** (không bàn đạp, kiểu xe tay ga) thuộc `Motorcycle` hoặc `Motorcyclist`.
- Xe đạp có chở hàng cồng kềnh: mask phủ cả hàng buộc vào xe.

### Pedestrian (8)
- Gồm: người đi bộ, đứng, ngồi (không ngồi trên phương tiện), chạy bộ, trẻ em, người dắt xe, người dùng xe lăn tự đẩy, người bán hàng rong đang đứng cạnh xe đẩy, người đang kéo xe kéo.
- Mask phủ người nhìn thấy, gồm trang phục, túi xách đeo, nón, ô che (nếu cầm trên người). Không phủ vật thể không gắn với người (xe đẩy lớn, vali để cạnh).
- Người ngồi ghế nhựa ở quán vỉa hè: **Pedestrian**.
- Người bế hoặc cõng trẻ em: **một instance** phủ cả hai.

### Sign (9)
- Gồm: biển báo hiệu đường bộ theo quy chuẩn (cấm, nguy hiểm, hiệu lệnh, chỉ dẫn, biển phụ), biển chỉ đường lớn (cổng giá long môn), biển tạm công trường, biển tên đường nếu là biển giao thông chính thức.
- Mask phủ **mặt biển**, không phủ cột, không phủ giá đỡ. Biển nhìn từ phía sau (lưng biển) vẫn gán.
- Nhiều biển trên cùng một cột: **mỗi tấm biển một instance**, kể cả biển phụ gắn bên dưới.
- Không gán: biển hiệu quảng cáo, biển cửa hàng, băng rôn, bảng giá, áp phích, biển số nhà.
- Biển nhỏ ở xa (dưới ngưỡng ở mục 2) bỏ qua.

### Traffic light (10)
- Gồm: hộp đèn tín hiệu xe cộ, đèn tín hiệu cho người đi bộ, đèn đếm ngược **gắn liền** trong cùng hộp.
- Mask phủ **cả hộp đèn** (khung và các bóng), không chỉ bóng đang sáng. Nhìn từ trước hay từ sau đều gán.
- Nhiều hộp đèn trên một cột: mỗi hộp một instance.
- Không gán: đèn xe, đèn trang trí, đèn LED quảng cáo, đèn chớp cảnh báo công trường nhỏ, đèn đường.

## 4. Các tình huống thường gặp và ca đặc biệt

| Tình huống | Cách gán |
|---|---|
| Xe máy chở nhiều người (2-3 người) | Một `Motorcyclist` cho cả xe và mọi người |
| Xe máy chở hàng cồng kềnh (lồng gà, tủ, cây) | Hàng buộc chắc vào xe: nằm trong mask. Hàng nhô xa, không rõ thuộc xe: không phủ |
| Người ngồi sau xe máy nhìn thấy rõ, xe che mất nửa | Vẫn một instance Motorcyclist |
| Xe máy đỗ thành dãy dài sát nhau (bãi xe) | Gán từng xe nếu tách được. Dãy quá dày, không tách nổi: gán các xe rõ nhất và bỏ phần không phân biệt được |
| Xe máy bị che gần hết (chỉ thấy bánh hoặc tay lái) | Gán nếu phần nhìn thấy vượt ngưỡng và nhận ra là xe máy. Không đủ thì bỏ |
| Người dắt xe máy bị hỏng, đứng cạnh | `Pedestrian` + `Motorcycle` (hai instance) |
| Người đang bước lên hoặc bước xuống xe | Đã ngồi trên yên: Motorcyclist. Chưa ngồi: Pedestrian + Motorcycle |
| Xe đạp buộc hoặc chở sau xe máy hoặc ô tô | Gán xe đạp (`Bicycle`) nếu rõ, xe chở là instance riêng |
| Xe máy chở trên thùng xe tải | Gán xe máy (`Motorcycle`) nếu rõ, xe tải là instance riêng, mask xe tải không phủ xe máy |
| Xe đầu kéo + container | Đầu kéo = `Truck`, container / rơ-moóc = `Other vehicle` |
| Xe buýt có quảng cáo dán toàn thân | Vẫn là `Bus`, mask phủ toàn bộ thân xe |
| Xe cứu thương, xe cảnh sát, xe cứu hỏa | Cứu thương dạng van: `Car`. Cứu hỏa: `Truck`. Xe cảnh sát thông thường: `Car` |
| Xe ủi, xe xúc, xe cẩu, máy công trình | `Other vehicle` |
| Xe ô tô điện | Vẫn là `Car` (phân loại theo hình dạng, không theo động cơ) |
| Xích lô, xe ba gác chở hàng, xe lôi | `Other vehicle`. Người ngồi hoặc đạp: gộp vào cùng instance |
| Phản chiếu ô tô trên kính tòa nhà hoặc vũng nước | Không gán |
| Áp phích quảng cáo có hình xe hoặc người | Không gán |
| Đèn giao thông nhìn từ phía sau | Vẫn gán |
| Cột đèn giao thông có hai hộp xếp chồng | Hai instance `Traffic light` |
| Biển báo bị che nửa bởi cành cây | Gán phần nhìn thấy nếu đủ ngưỡng |
| Đêm tối, chỉ thấy đèn xe sáng | Gán nếu nhận ra hình thể phương tiện. Chỉ thấy mỗi đèn thì không gán |
| Ảnh mờ do chuyển động | Gán nếu nhận ra được, mask theo hình thể nhìn thấy |
| Người trên xe lăn do người khác đẩy | Người ngồi xe lăn: `Pedestrian`; người đẩy: `Pedestrian`; xe lăn không gán |
| Đám đông (chợ, lễ hội) | Mỗi người rõ là một instance. Đám quá dày không tách được: gán người ở hàng trước, bỏ phần còn lại |
| Cảnh có động vật | **Không gán** (không có lớp tương ứng) |

## 5. Cây quyết định khi phân vân

1. Là vật thể **đang di chuyển hoặc đỗ trên đường**? Nếu không (cố định bên đường, ví dụ biển, đèn) thì xem Sign hoặc Traffic light.
2. Có người ngồi trên xe 2 bánh?
   - Xe máy: có người là `Motorcyclist`; không người là `Motorcycle`.
   - Xe đạp: có người là `Cyclist`; không người là `Bicycle`.
3. Xe 4 bánh trở lên: nhỏ và chở được dưới khoảng 9 người là `Car`; chở nhiều, kiểu xe buýt hoặc xe khách là `Bus`; xe tải hoặc đầu kéo là `Truck`; còn lại (ba gác, rơ-moóc, công trình) là `Other vehicle`.
4. Vẫn chưa quyết được: chọn lớp gần nhất và ghi lại quyết định trong tài liệu quy ước của đội.

## 6. Lỗi thường gặp (hãy tự kiểm tra trước khi huấn luyện)

- Quên gán đối tượng nhỏ nhưng vẫn vượt ngưỡng (đặc biệt Pedestrian ở xa, Sign, Traffic light).
- Mask Motorcyclist chỉ phủ người mà không phủ xe (hoặc ngược lại).
- Mask của hai đối tượng chồng lên nhau.
- Dùng sai lớp cho xe đạp điện, xe ba gác, xe 16 chỗ.
- Mask quá rộng, tràn nền hoặc bóng đổ.
- Gán biển quảng cáo hoặc ảnh trên áp phích.
- Một đối tượng bị tách thành hai instance, hoặc hai đối tượng bị gộp thành một.
