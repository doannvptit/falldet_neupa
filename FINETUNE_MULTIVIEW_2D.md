# Fall detection từ pose ảnh 2D, train đa góc nhìn

Pipeline inference:

`RGB → landmark X/Y trên ảnh → map NTU25 → center/scale 2D → binary NeuroPath`

Classifier không dùng độ sâu, PCA, hay `pose_world_landmarks`. Đầu vào vẫn
`[B,3,64,25,2]` để giữ kiến trúc pretrained; kênh Z luôn bằng 0. API trực tiếp
`BinaryClassifier.score(poses)` nhận mảng `[T,25,2]` pixel X/Y khi dùng config
multiview. Nếu truyền `[T,25,3]`, cột thứ ba bị bỏ qua hoàn toàn, kể cả NaN/inf.

MediaPipe hiện là bộ trích landmark ảnh 33 khớp để map sang 25. Nó có khả năng
xuất 3D trong nội bộ model, nhưng code ở chế độ này chỉ đọc `x/y/visibility`,
không đọc trường `z` hay kết quả world; không có bước nâng pose 2D lên 3D.
Có thể thay bộ pose bằng estimator 2D khác nếu map về đúng 25 khớp và đúng thứ
tự anatomical trái/phải. Tọa độ ảnh chuẩn hóa phải đổi về pixel theo width/height
trước khi center/scale để tránh méo aspect ratio.

## Dữ liệu train

Nguồn `data/ntu_fall_camera_xyz` có 40.091 train và 16.487 test, tương ứng 671 và
275 mẫu fall. Dùng skeleton 3D hiện có để mô phỏng camera **chỉ khi tạo dữ liệu
train/test tổng hợp**, không cần dựng 3D từ video ở inference.
Dataset này được tạo lại từ raw skeleton với `--coordinate-mode camera_xyz`:
chỉ center/hold-fill, giữ trục Y camera hướng lên, không dùng các phép xoay
theo thân người của dataset 3D/PCA cũ. Feeder kiểm tra metadata để tránh
dùng nhầm dữ liệu đã bị xoay.

Camera giữ cố định trong cả clip để không tạo chuyển động giả:

Các tham số góc/khoảng cách giữ cố định trong hệ pose đã center root theo
frame. Vì vậy đây là augmentation hình học của pose, không phải renderer
camera tĩnh đầy đủ trong không gian cảnh; vị trí toàn cục của root đã bị bỏ.

| Nhóm camera train | Tỷ lệ lấy mẫu | Góc nâng camera trên mặt phẳng ngang | Azimuth |
|---|---:|---|---|
| Trên nhìn xuống | 50% | 60–88° | Ngẫu nhiên toàn vòng |
| Bên trái | 20% | 0–35° | 0° ±35° |
| Bên phải | 20% | 0–35° | 180° ±35° |
| Góc xiên khác | 10% | 0–60° | Ngẫu nhiên toàn vòng |

Các góc tính trong hệ camera NTU gốc, Y hướng lên, X/Z là mặt phẳng ngang.
Camera nguồn được giả định gần thẳng đứng với mặt đất; không có hiệu chuẩn
gravity/extrinsic chính xác cho từng clip. Dùng cả phối cảnh (70%, khoảng cách
3–7 mét tương đối với root) và phép chiếu trực giao (30%). Augmentation thêm roll trong mặt phẳng
±0,3 rad, mirror, jitter và mất ngắn một số khớp.

Chuẩn hóa ảnh chỉ dùng X/Y: center tại spine-mid theo frame, chia cho một
bán kính RMS cơ thể trung vị trong cửa sổ. Không xoay thân người về phương
thẳng đứng khi inference vì hướng/nghiêng trên ảnh có ích cho nhận diện ngã.
Góc nhìn từ trên cao có thể làm các chi chồng lên nhau; mô phỏng không thay
thế được lỗi/occlusion của estimator trong video thật.

## Train và kiểm chứng

Config: `config/fall-detection/image2d_multiview_2epoch.yaml`.
Khởi tạo từ binary 3D epoch 2 trước đó, không khởi tạo từ checkpoint PCA.
Fine-tune 2 epoch, LR 0,003, batch 64, forward batch 16 trên RTX 3060.
Mỗi epoch dùng 671 fall + 5.368 non-fall; negatives được lấy lại mỗi epoch.

Mỗi clip test được chiếu cố định theo cả 3 camera: overhead (azimuth 45°,
elevation 80°), side_left (0°,15°), side_right (180°,15°), khoảng cách 5.
Subset theo dõi gồm 2.475 clip ×3 =7.425 phép chiếu. Full test có
16.487 clip ×3 =49.461 phép chiếu; **không phải 49.461 clip độc lập**.
Chọn checkpoint bằng fall F1 tổng hợp trên subset theo dõi.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/falldet-matplotlib \
  .venv/bin/python main.py --config config/fall-detection/image2d_multiview_2epoch.yaml
```

Đánh giá toàn bộ test, kèm metrics từng camera:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/falldet-matplotlib \
  .venv/bin/python eval_fall.py \
  --config config/fall-detection/image2d_multiview_2epoch.yaml --batch 32 \
  --output-dir work_dir/fall/image2d_multiview_2epoch/full_test
```

Chạy video bằng checkpoint tốt nhất của đúng config:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python infer_fall_video.py \
  --config config/fall-detection/image2d_multiview_2epoch.yaml \
  --video-dir ../PdPdFall-ONNX/videos \
  --output-dir outputs/mediapipe_fall_image2d
```

Có thể truyền `--weights` để chọn epoch cụ thể. Bản 3D/PCA và các kết quả trước
được lưu riêng. Xem kết quả thực chạy trong
`work_dir/fall/image2d_multiview_2epoch/RESULTS.md`.

Gói checkpoint và cấu hình inference được lưu ở `models/fall_image2d_multiview/`.
Cấu hình trong gói không trỏ tới dữ liệu NTU hay PCA:

```bash
.venv/bin/python infer_fall_video.py \
  --config models/fall_image2d_multiview/inference.yaml \
  --video ../PdPdFall-ONNX/videos/20240912_101331.mp4 \
  --output-dir outputs/image2d_single
```

Nếu đã có pose X/Y 25 khớp từ estimator 2D, có thể bỏ bước MediaPipe và gọi
`BinaryClassifier.score()` trực tiếp trên một cửa sổ pose. Chỉ cần code model,
checkpoint và cấu hình inference; không cần raw skeleton hoặc dataset 3D.

Metrics trên camera mô phỏng chỉ đo bài toán tổng hợp. Để kết luận chất lượng
góc trên xuống/hai bên trong thực tế, cần video RGB có nhãn ở các góc đó và
tốt hơn là fine-tune thêm trên pose 2D trích từ chính video. Các video hiện có
chưa có nhãn chuẩn theo người/thời gian để tính precision/recall/F1 thực tế.
