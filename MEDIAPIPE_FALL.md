# Test video RGB bằng checkpoint binary đã finetune

**Mặc định mới:** classifier dùng **pose ảnh 2D đa góc nhìn, Z=0**, không dùng
world pose/PCA. Xem [hướng dẫn mới](FINETUNE_MULTIVIEW_2D.md). Các kết quả và
phần mô tả 3D dưới đây thuộc lượt thử trước. Muốn chạy lại bản 3D, truyền
`--config config/fall-detection/verify_2epoch.yaml`; CLI tự chọn best weights
của đúng config, hoặc dùng `--weights` để chỉ định rõ.

Đã triển khai luồng `video RGB → MediaPipe Pose Landmarker → tracking →
33 landmark thành NTU25 → preprocessing khi train → NeuroPath binary → video overlay`.
Entry point: `infer_fall_video.py`. Trọng số mặc định là
`work_dir/fall/verify_3epoch/train/best/weights-2-17090.pt`, config
`config/fall-detection/verify_2epoch.yaml`. Nhãn `0=non-fall`, `1=fall`.

## Cài đặt và chạy

Chạy từ thư mục `poc`, dùng môi trường `.venv` đã có của bài toán finetune.
Script cần terminal truy cập được PyPI và Google Storage; dùng `uv`, `curl`
và FFmpeg đã có trên máy:

```bash
bash scripts/setup_video_fall.sh
```

Script cài dependency trong `requirements-video.txt`, tải
`models/pose_landmarker_full.task`, rồi kiểm tra dependency và checkpoint.
Dùng trực tiếp `.venv/bin/python` cho các lệnh bên dưới để giữ các dependency
video vừa cài; không cần chạy lại đồng bộ môi trường bằng `uv run`.

Thử clip ngắn trước:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python infer_fall_video.py \
  --video ../PdPdFall-ONNX/videos/20240912_101331.mp4 \
  --output-dir outputs/mediapipe_smoke
```

Chạy tất cả video:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python infer_fall_video.py \
  --video-dir ../PdPdFall-ONNX/videos
```

Classifier mặc định `--device auto`: dùng CUDA nếu PyTorch truy cập được,
ngược lại dùng CPU. Có thể yêu cầu `--device cuda:0` hoặc `--device cpu`.
MediaPipe trong triển khai này dùng CPU. `nvidia-smi` hoạt động trên host
không bảo đảm phiên sandbox có quyền truy cập thiết bị GPU.

## Kết quả xuất ra

Mỗi video có một thư mục `outputs/mediapipe_fall/<tên video>/`:

| File | Nội dung |
|---|---|
| `annotated.mp4` | H.264 với skeleton, track ID, FALL/NON-FALL, xác suất; không kèm audio |
| `poses.npz` | Landmark ảnh 33, world 33, NTU25, visibility, frame và track ID |
| `extraction.json` | FPS, thông tin nguồn, phiên bản MediaPipe, SHA256 model pose, tham số trích xuất |
| `windows.csv` | Khoảng frame/thời gian, độ phủ pose, xác suất fall và nhãn mỗi cửa sổ |
| `frames.jsonl` | Trạng thái và xác suất từng người theo từng frame |
| `summary.json` | Số frame, track, cửa sổ và cấu hình inference; chưa có ground truth |
| `ffmpeg.log` | Lỗi encoder nếu có |

Thư mục đầu ra còn có `summary.json` tổng hợp các video. `fall_windows` đếm
cửa sổ dự đoán fall, không phải số sự kiện ngã độc lập.

Sau lần chạy đầu, có thể thử ngưỡng khác mà không trích pose lại:

```bash
.venv/bin/python infer_fall_video.py \
  --video-dir ../PdPdFall-ONNX/videos --reuse-poses --threshold 0.7
```

Giữ nguyên thư mục đầu ra và các tham số trích xuất khi tái sử dụng pose.
Code kiểm tra đường dẫn, kích thước và thời điểm sửa video, cùng tham số
trích xuất; nếu thay đổi thì chạy lại không có `--reuse-poses`.

## Cửa sổ và giới hạn

- Mỗi người được phân loại riêng, tối đa 2 pose/frame mặc định. Tracking
  ghép tâm hông giữa các frame, có thể nhầm ID khi người che nhau hoặc đi sát.
- Mặc định dùng tối đa 64 frame nguồn, bước 8 frame, tối thiểu 16 frame có pose
  và độ phủ 60%. Cửa sổ được crop/resize về 64 frame theo feeder của model.
  Có xử lý cửa sổ cuối video, kể cả clip ngắn hơn 64 frame.
- Trước khi đủ dữ liệu hiển thị `WARMUP`; frame không có pose hiển thị
  `NO POSE`. Kết quả chỉ xuất hiện sau frame kết thúc cửa sổ tương ứng.
- MediaPipe ước lượng 33 landmark, gồm tọa độ world tương đối với tâm hông.
  Adapter tạo 25 khớp theo thứ tự NTU; spine/neck/hand có nội suy nên đây là
  skeleton xấp xỉ, không phải dữ liệu Kinect 25 khớp đo trực tiếp.
- Preprocessing giữ cách center/rotate, float16 round-trip và crop/resize
  của pipeline train. Model nhận `[B,3,64,25,2]`; dùng output fused 2 lớp,
  `p_fall = softmax(logits)[1]`, ngưỡng mặc định 0,5.
- Chất lượng trên NTU không đại diện chất lượng trên video RGB. Cần nhãn
  người và khoảng thời gian fall/non-fall để đo precision/recall/F1 thực tế.

Tham khảo API và hệ tọa độ của
[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
Mapping chi tiết trong `video_fall/adapter.py`.

## Đã kiểm tra trong phiên triển khai

- 10 test đạt: mapping/visibility, tracking, cửa sổ và mất frame, cache pose,
  overlay không dùng dự đoán tương lai và xuất H.264 bằng FFmpeg.
- Input tạo từ mẫu NTU gốc khớp từng phần tử với input đã lưu của pipeline
  train/test. Checkpoint epoch 2 chạy CPU trên mẫu fall này cho `p_fall=1.0`.
- Sau khi mở permission, đã cài MediaPipe 0.10.35, OpenCV 5.0.0.93 và tải
  model Pose Landmarker Full thành công. PyTorch nhận RTX 3060; classifier
  chạy `cuda:0`, MediaPipe dùng CPU. Cả 10 test đạt trong chính `.venv`.
- Đã chạy video RGB thật; kết quả chi tiết và kiểm tra artifact được lưu tại
  `outputs/mediapipe_fall/RUN_REPORT.md`. Hai clip ngã ngắn đầu tiên không
  phát hiện FALL ở ngưỡng 0,5; cần cải thiện/đánh giá adapter trên dữ liệu RGB.

Test tự động, sau khi cài dependency:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v
```
