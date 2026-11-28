# Hồ sơ chạy lại — team21-mask-roi

Mô hình điểm cao nhất của Đội 21 đạt **27,5365 mask mAP@[.50:.95]** (27,54/100 khi làm tròn) trên validation khóa cứng 140 ảnh. Hồ sơ bên dưới phân biệt replay chấm ONNX với huấn luyện lại toàn bộ lineage.

## Hồ sơ đã lưu trong repo

| Tệp | Vai trò |
|---|---|
| `run_report.json` | Cấu hình, split, thống kê lớp, lineage checkpoint, checksum và metric của lần chạy chọn. |
| `resources.json` | Revision mã, SHA notebook, SHA gói dữ liệu và thông tin tài nguyên của lượt chạy. |
| `validation-manifest.json` | Danh sách 140 ảnh val cùng SHA ảnh/nhãn; SHA-256 file là `23c92420f1742f9d6091c09930bb43e055558064b604a2b61291b62eaeeb2013`. |
| `../../../notebooks/archive/train_and_export_v2.ipynb` | Notebook đã dùng; SHA-256 `9cf720e6cad44b8dec33e9a6fafc8425ce74e61d1fb37da65584b2ac9d4a73c5`. |
| `../../../artifacts/team21-mask-roi/submission.zip` | ZIP nộp chứa `model.onnx`; SHA-256 `2717f063091799facee8526bab18e4fc686424109d37d1b778588e4c42b3070e`. |

Run report ghi fingerprint cây dữ liệu đã chuẩn bị là `7e8e27dd018b71d8b1c9a9835680ec6088b0d707950883e6cd73276700936b10`. Gói nguồn `e3-triage-v1/dataset.zip` dùng để dựng lại tập này có SHA-256 `33362fd1aee1760da38f0f67e41a97c61dcc6aed6c10561bfbf9ef058755c2ef`.

## 8 gói dữ liệu của 9 checkpoint nguồn

Các ZIP được giữ trong kho dữ liệu của đội ở `../team-data/overnight/<tên-gói>/dataset.zip`, tính từ thư mục gốc repo. Chúng không nằm trong Git vì tổng dung lượng khoảng **2,57 GiB** và gói có điều khoản nguồn khác nhau. Khi ban tổ chức yêu cầu, bàn giao đúng gói theo tên và kiểm lại SHA trước khi chạy.

| Gói dữ liệu | Dùng bởi checkpoint | Dung lượng (byte) | SHA-256 của ZIP |
|---|---|---:|---|
| `e3-triage-v1` | `team21-qc20`, `team21-sgd20` | 198857764 | `33362fd1aee1760da38f0f67e41a97c61dcc6aed6c10561bfbf9ef058755c2ef` |
| `team21-btc-v1` | `team21-btc20` | 198861553 | `7141c5527ec6bb2a86bb897f336435bcfd2a0be705efb6909e9dd92cfc527010` |
| `team21-road-v1` | `team21-road20` | 204947536 | `155b1d3cc81ea8b87c6123795a02702eb847b3807c85b76d4d21e496225e3252` |
| `team21-signfaces-v1` | `team21-full` | 198857893 | `935c2e808540529595eb21fc175a79df35f7f2050615b7cc480ef047a3935ba0` |
| `e3-vnrepair-v1` | `team21-vnwide20` | 422979797 | `03b99ac7680044dfb2ae2b1e22839d78fcfc20cf853e6c33c39b2b4d5e52daa2` |
| `team21-wideqc-v1` | `team21-wideqc20` | 448544012 | `cd96a38e4659800d5ed0ad0bcd5fa3d82368170ac0d750360d11d341d6fef244` |
| `team21-vistas-extra-v1` | `team21-vistas20` | 518406662 | `5e3460b7d794fdea49ea0654870ba60d26deb093907141107972717b430ab371` |
| `team21-paste-v1` | `team21-paste20` | 566257189 | `e1ba149c68ce94b339d6aae935aed4708e9b27770bdbfbf663a39e87ab2f83d2` |

Các `best.pt` của chín checkpoint, checkpoint `team21-mean-paste-bn` và kết quả từng lượt cũng được giữ trong các thư mục cùng tên dưới kho `team-data/overnight/`. Bản chọn tiếp tục từ `team21-mean-paste-bn`, BN refresh trên 168 ảnh train, rồi fit ROI mask chỉ trên nhãn train; report ghi chọn epoch 7.

## Replay điểm ONNX trên CPU

Chạy từ thư mục gốc repo sau khi kho `team-data` được cấp cùng máy. `--out` cần trỏ tới một thư mục mới, chưa tồn tại.

```bash
python scripts/replay_submission.py --notebook notebooks/archive/train_and_export_v2.ipynb --dataset ../team-data/overnight/e3-triage-v1/dataset.zip --submission artifacts/team21-mask-roi/submission.zip --report reports/replay/team21-mask-roi/run_report.json --resources reports/replay/team21-mask-roi/resources.json --validation reports/replay/team21-mask-roi/validation-manifest.json --out /path/to/new-replay-output
```

Script xác nhận SHA notebook, ZIP dữ liệu, model và manifest validation trước khi dựng dữ liệu notebook rồi chấm ONNX. Huấn luyện lại toàn bộ ensemble cần đủ tám ZIP và `best.pt` của chín checkpoint nguồn được liệt kê ở trên.
