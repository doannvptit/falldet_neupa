# Fine-tune pose 2D với Z=0, giữ tối đa phương sai 3D

Đây là phương án trung gian đã chạy xong 2 epoch. Theo yêu cầu mới không cần
pose 3D khi inference, dùng [pipeline image 2D đa góc nhìn](FINETUNE_MULTIVIEW_2D.md)
và config `image2d_multiview_2epoch.yaml` cho lượt train tiếp theo.

Yêu cầu: đưa thông tin pose 3D vào hai chiều X/Y, đầu vào classifier có Z=0.
Dùng một phép chiếu PCA cố định học từ tập train, sau chuẩn hóa NTU như bản 3D.
Trong các phép chiếu tuyến tính trực giao xuống hai chiều, PCA tối thiểu hóa
sai số tái tạo bình phương trung bình trên các điểm dùng để fit. Điều này
không bảo đảm giữ mọi thông tin hữu ích cho phân loại ngã.

## Phép chiếu

- Fit trên toàn bộ 40.091 mẫu train, 92.396.344 điểm khớp khác zero; không fit
  trên test hoặc video RGB. Không đổi PCA theo clip/frame nên giữ hệ trục cố định.
- Giữ **92,3859% phương sai tọa độ**; mất 7,6141%. Nếu chỉ xóa Z sau chuẩn hóa
  3D cũ thì giữ 72,0724% phương sai trên cùng tập điểm.
- `XY_mới = (XYZ_đã_chuẩn_hóa - mean_train) @ components.T`, `Z_mới = 0`.
  Joint/frame zero padding vẫn giữ zero.
- Không cần tạo bản sao dataset: feeder chiếu trực tiếp các mảng mmap gốc.
- Input model vẫn `[B,3,64,25,2]` để dùng trọng số pretrained. Z bằng 0 cả
  sau crop/resize, phép quay augmentation trong mặt phẳng XY, và khi suy luận.
- Đây là **biểu diễn 2D của pose 3D**, không phải tọa độ pixel 2D. Pipeline
  video vẫn lấy pose world 3D của MediaPipe trước khi chuẩn hóa và chiếu PCA.

Artifact bắt buộc:

- `data/ntu120_fall_pca2d/projection.npz`: mean và hai vector cơ sở.
- `data/ntu120_fall_pca2d/projection.json`: eigenvalues, tỷ lệ giữ lại và nguồn fit.
- `config/fall-detection/pca2d_zero_z_2epoch.yaml`: cấu hình train/test/inference.
- `work_dir/fall/pca2d_zero_z_2epoch/train/`: checkpoint, log và kết quả.

Không thay PCA khi dùng checkpoint đã train: PCA là một phần của model pipeline.

## Fine-tune

Khởi tạo từ checkpoint binary 3D epoch 2
`work_dir/fall/verify_3epoch/train/best/weights-2-17090.pt`, giữ cả hai head.
Fine-tune thêm 2 epoch trên GPU, LR 0,001 cho toàn model, SGD Nesterov,
batch 64 (4 lượt forward 16 để giới hạn VRAM), class weight `[0.25,1.0]`.
Mỗi epoch chọn 671 fall + 5.368 non-fall; train loader bỏ batch cuối chưa đủ.
Đánh giá mỗi epoch trên cùng subset 275 fall + 2.200 non-fall như lần 3D trước,
chọn checkpoint theo fall F1. Toàn bộ split test có 16.487 mẫu, gồm 275 fall.

Chạy từ thư mục `poc`:

```bash
# Chỉ fit một lần; script từ chối ghi đè projection đã có.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/fit_pose_pca.py

# Khi tái chạy, dùng work_dir mới để tách khỏi kết quả hiện có.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/falldet-matplotlib \
  .venv/bin/python main.py --config config/fall-detection/pca2d_zero_z_2epoch.yaml
```

Test toàn bộ skeleton test:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/falldet-matplotlib \
  .venv/bin/python eval_fall.py \
  --config config/fall-detection/pca2d_zero_z_2epoch.yaml --batch 32 \
  --output-dir work_dir/fall/pca2d_zero_z_2epoch/full_test
```

Test video với checkpoint mới: truyền `--config` như trên và `--weights`
trỏ vào file `weights-*.pt` trong thư mục `best` mới; giữ `--output-dir` riêng
để so sánh với kết quả 3D. Có thể dùng lại cache pose của lần chạy trước vì PCA
chỉ áp dụng sau trích xuất pose. Chi tiết kết quả thực chạy sẽ có trong
`work_dir/fall/pca2d_zero_z_2epoch/RESULTS.md`.
