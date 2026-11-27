# Rà soát xích lô và xe ba bánh

Tiến phụ trách [CVAT Job 13](http://34.58.69.170:8080/tasks/13/jobs/13). Trong 58 ảnh, 150 polygon đã được nạp sẵn. Sửa đường biên theo phần xe thật sự nhìn thấy và thêm mọi instance còn thiếu thuộc 11 lớp. Đánh dấu ảnh hoàn tất sau khi một người khác kiểm lại. Ghi rõ số xe ba bánh đã được chấp nhận trước khi xuất bộ train.

24 vị trí dưới đây có mask SAM nhưng polygon đơn mất quá nhiều chi tiết. Tìm đúng tên ảnh trong CVAT. Tọa độ là x trái, y trên, x phải, y dưới, tính bằng pixel trên ảnh đã tải. Dùng box để tìm xe và vẽ theo đường biên thật.

| Ảnh | Loại theo nguồn | Vùng cần kiểm tra |
|---|---|---|
| bd_mixed_1.jpg | cng | 969, 270, 1092, 314 |
| bd_mixed_16.jpg | rickshaw | 808, 499, 1280, 916 |
| bd_mixed_18.jpg | rickshaw | 843, 405, 960, 711 |
| bd_mixed_3.jpg | rickshaw | 277, 133, 375, 211 |
| bd_mixed_4.jpg | rickshaw | 158, 110, 302, 213 |
| bd_mixed_4.jpg | rickshaw | 316, 112, 453, 209 |
| bd_mixed_4.jpg | rickshaw | 318, 67, 436, 162 |
| bd_mixed_45.jpg | rickshaw | 337, 919, 790, 1280 |
| bd_mixed_49.jpg | rickshaw | 817, 359, 847, 421 |
| bd_mixed_50.jpg | cng | 428, 504, 452, 528 |
| bd_mixed_52.jpg | cng | 383, 558, 407, 599 |
| bd_mixed_62.jpg | cng | 1067, 219, 1143, 273 |
| bd_mixed_62.jpg | cng | 153, 219, 297, 303 |
| bd_mixed_62.jpg | cng | 301, 220, 425, 275 |
| bd_mixed_63.jpg | cng | 961, 189, 1109, 208 |
| bd_mixed_65.jpg | cng | 1, 116, 219, 275 |
| bd_mixed_66.jpg | cng | 902, 344, 998, 373 |
| bd_mixed_66.jpg | cng | 253, 334, 355, 365 |
| bd_mixed_7.jpg | rickshaw | 30, 37, 54, 54 |
| bd_mixed_7.jpg | rickshaw | 157, 32, 174, 57 |
| bd_mixed_70.jpg | cng | 110, 126, 165, 163 |
| bd_mixed_75.jpg | cng | 847, 6, 870, 34 |
| bd_mixed_79.jpg | rickshaw | 633, 435, 958, 775 |
| bd_mixed_84.jpg | rickshaw | 675, 719, 768, 788 |
