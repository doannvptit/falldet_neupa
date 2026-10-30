# Benchmark hiệu năng — pipeline ONNX phát hiện ngã

Đo trên máy đang chạy POC, dùng đúng code runtime của `poc_inference_onnx`
(YOLOX-s → RTMPose-m Halpe26 → NTU25 → fall ONNX). Mọi số trong báo cáo lấy từ
`scripts/benchmark_pipeline.py`, lưu tại `outputs/benchmark/<device>/results.json`
và `micro.json`. Không có con số nào ước lượng.

## 1. Tóm tắt

| | CUDA (RTX 3060) | CPU (4 luồng) |
|---|---:|---:|
| Thời gian xử lý 4 video (1058 frame, 1080p) | **52.4 s** | **212.9 s** |
| Trung bình mỗi frame | 49.5 ms | 201.3 ms |
| Throughput trung bình (có overlay) | **18.4 FPS** | 4.8 FPS |
| Tốc độ so với thời gian thực (video 30 FPS) | 0.46–0.73× | 0.14–0.18× |
| Throughput **không overlay** (`--no-render`) | **33.2 FPS — 1.10× realtime** | 5.4 FPS (0.18× realtime) |
| Bộ nhớ đỉnh (RSS) | 1.44 GB | 1.57 GB |

Nút thắt cổ chai: **YOLOX-s** (38% GPU / 67% CPU) và **encode overlay**
(31% GPU / 8% CPU). Bộ phân loại ngã gần như miễn phí (1.7 ms/frame trên GPU).

## 2. Môi trường đo

| Hạng mục | Giá trị |
|---|---|
| CPU | Intel Xeon E5-2678 v3 @ 2.50 GHz, 12 nhân / 24 luồng |
| RAM | 46 GB |
| GPU | NVIDIA GeForce RTX 3060 12 GB, compute 8.6, driver 580.178.04, CUDA 13.0 |
| OS | Linux 6.8.0-138-generic (Ubuntu 22.04), glibc 2.35 |
| Python | 3.12.13 |
| ONNX Runtime | 1.31.0 (provider: CUDA / TensorRT / CPU) |
| NumPy / OpenCV | 2.4.6 / 5.0.0 |
| ffmpeg | 4.4.2 (libx264, pipe rawvideo) |
| GPU khi đo | SM ~2000 MHz, P2, 70–74% utilization, ~102 W |

### Model ONNX dùng trong pipeline

| Model | File | Kích thước | Input | Output | Ghi chú |
|---|---|---:|---|---|---|
| YOLOX-s (human-art) | `models/yolox_s.onnx` | 35.99 MB | `[1,3,640,640]` | `dets [1,N,5]`, `labels [1,N]` | SHA256 `332e09ea9696…` |
| RTMPose-m Halpe26 | `models/rtmpose_m_halpe26.onnx` | 55.69 MB | `[batch,3,256,192]` | `simcc_x [b,26,384]`, `simcc_y [b,26,512]` | SHA256 `26f3a19e6130…` |
| Fall EGGCN (fusion) | `models/fall_image2d.onnx` | 15.70 MB | `skeleton [1,3,64,25,2]` | `logits [1,2]` | opset 17, 2 659 699 tham số, SHA256 `64ed11cadb02…` |

Tổng bundle model: **107.4 MB**, không cần PyTorch.

## 3. Quy trình đo

- `scripts/benchmark_pipeline.py` chạy đúng đường đi production: decode → YOLOX →
  RTMPose → map Halpe26→NTU25 → tracker → cửa sổ nhân quả → tiền xử lý → ONNX →
  overlay H.264, có bộ đếm thời gian `perf_counter` cho từng khâu.
- Micro-benchmark: warm-up 3 lần, lặp 20–30 lần, báo median/mean/p95.
- Cold start: tiến trình con mới, đo thời gian tạo session và lần inference đầu.
- Thread scaling: CPU, đo cùng một đầu vào với 1/2/4/8/16 luồng intra-op.
- Mọi lần chạy lại cho kết quả trong báo cáo này:

```bash
cd /mnt/doannv/home/lumi/luvox-project/falldet/poc_inference_onnx
.venv/bin/python scripts/benchmark_pipeline.py --device cuda --threads 4 --repeat 20
.venv/bin/python scripts/benchmark_pipeline.py --device cpu  --threads 4 --repeat 20
.venv/bin/python scripts/benchmark_pipeline.py --micro-only --device cuda --repeat 30
```

## 4. Độ trễ từng thành phần (median, ms)

| Thành phần | CPU (4 luồng) | CUDA | Tăng tốc GPU | Ghi chú |
|---|---:|---:|---:|---|
| Decode 1 frame (1920×1080 H.264) | 1.58 | 1.50 | 1.1× | bị giới hạn bởi OpenCV/FFmpeg, không dùng GPU |
| **YOLOX-s detect** (1 frame) | 120.64 | 18.89 | **6.4×** | tổng: letterbox 0.95 + ONNX + NMS 0.01 |
| ↳ letterbox resize 1920×1080→640×640 | 0.95 | 0.93 | 1.0× | luôn CPU |
| ↳ YOLOX ONNX run | 120.43 | 14.79 | 8.1× | tensor 4.92 MB/frame |
| ↳ NMS + lọc ngưỡng | 0.010 | 0.012 | — | NMS đã nhúng trong graph |
| **RTMPose** (1 người, 1 crop) | 25.09 | 5.92 | **4.2×** | tổng: affine 2.4 + ONNX + simcc 0.07 |
| ↳ affine crop 1920×1080→256×192 | 2.38 | 2.42 | 1.0× | `warpAffine` trên CPU |
| ↳ RTMPose ONNX run | 21.41 | 2.77 | 7.7× | tensor 0.59 MB/crop |
| ↳ decode simcc → pixel XY | 0.068 | 0.073 | — | |
| Map Halpe26→NTU25 (25 khớp) | 0.19 | 0.22 | 0.9× | thuần NumPy |
| Tracker update (1 người) | 0.006 | 0.007 | — | thuần NumPy |
| Tiền xử lý legacy (64→tensor) | 0.577 | 0.562 | 1.0× | NumPy |
| Tiền xử lý linear (nội suy 64 pose) | 13.66 | 14.80 | 0.9× | vòng lặp Python theo khớp |
| **Fall ONNX** (1 cửa sổ) | 146.53 | 9.23 | **15.9×** | logits + softmax |
| Cả `score()` (tiền xử lý + ONNX) | 148.86 | 9.83 | 15.1× | |

## 5. End-to-end theo video

### 5.1 CUDA (`--device cuda`, 4 luồng)

| Video | Frame | Người | Cửa sổ | Tổng | ms/frame | Decode | Detect | Pose | Map+track | Classify | Overlay | FPS | Realtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240912_101331 | 56 | 53 | 6 | 4.09 s | 73.1 | 2.6 | 29.7 | 8.7 | 0.39 | 10.0 | 21.6 | 13.7 | 0.46 |
| 20240912_101427 | 57 | 55 | 7 | 2.92 s | 51.2 | 2.4 | 18.8 | 6.3 | 0.42 | 1.2 | 21.9 | 19.5 | 0.65 |
| demo | 282 | 421 | 55 | 15.28 s | 54.2 | 7.9 | 18.4 | 9.8 | 0.56 | 2.0 | 15.4 | 18.5 | 0.62 |
| fall_0002 | 663 | 480 | 59 | 30.12 s | 45.4 | 7.2 | 18.2 | 4.8 | 0.31 | 0.9 | 14.0 | 22.0 | 0.73 |
| **Tổng** | **1058** | 1009 | 127 | **52.42 s** | **49.5** | **6.9 (13.9%)** | **18.9 (38.2%)** | **6.4 (13.0%)** | **0.39 (0.8%)** | **1.68 (3.4%)** | **15.2 (30.7%)** | **18.4** | — |

Video đầu tiên hấp thụ **cold start của CUDA (~1.2 s)**; bỏ video đó thì detect
còn 18.3 ms/frame. Overlay là encode H.264 1080p — không cần thiết khi chạy
thật (chỉ để xem lại).

### 5.2 CPU (`--device cpu`, 4 luồng)

| Video | Frame | Người | Cửa sổ | Tổng | ms/frame | Decode | Detect | Pose | Map+track | Classify | Overlay | FPS | Realtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240912_101331 | 56 | 53 | 6 | 11.60 s | 207.1 | 3.4 | 134.7 | 30.5 | 0.43 | 17.3 | 20.8 | 4.8 | 0.16 |
| 20240912_101427 | 57 | 55 | 7 | 11.71 s | 205.4 | 2.3 | 135.1 | 29.3 | 0.38 | 18.0 | 20.2 | 4.9 | 0.16 |
| demo | 282 | 421 | 55 | 65.50 s | 232.3 | 2.0 | 136.8 | 48.5 | 0.52 | 28.8 | 15.6 | 4.3 | 0.14 |
| fall_0002 | 663 | 480 | 59 | 124.14 s | 187.2 | 2.1 | 133.9 | 24.1 | 0.32 | 13.0 | 13.8 | 5.3 | 0.18 |
| **Tổng** | **1058** | 1009 | 127 | **212.94 s** | **201.3** | **2.2 (1.1%)** | **134.7 (66.9%)** | **31.2 (15.5%)** | **0.38 (0.2%)** | **17.7 (8.8%)** | **15.0 (7.5%)** | **4.8** | — |

Demo chậm hơn (pose 48.5 ms/frame) vì có 421 lượt người ở 282 frame → 1.5 crop/frame.

## 6. Phân tích

**Bottleneck theo thứ tự**

1. **YOLOX-s**: 18.9 ms/frame trên GPU, 134.7 ms/frame trên CPU. Trên GPU mức
   dùng ~72% util, ~2 GHz — đây là chi phí tính toán thật của graph detector
   (không phải nghẽn scheduler). GPU chỉ nhanh hơn CPU 6.4×.
2. **Encode overlay**: 14–22 ms/frame, **30.7% thời gian trên GPU**. Đây là chi phí
   trình diễn, không phải chi phí phát hiện. Chạy thật với `--no-render` đo được
   **33.2 FPS (1.10× realtime)** trên GPU.
3. **RTMPose**: 6.4 ms/frame trên GPU (mỗi người 5.9 ms). Batching 2 người vào
   một lần gọi (ONNX RTMPose cho phép batch động) đo được **11.6 ms → 9.5 ms**
   cho 2 crop (−18%), chưa triển khai trong runtime.
4. **Phân loại ngã**: 1.7 ms/frame (GPU) và 17.7 ms/frame (CPU). Rẻ nhất trong
   pipeline; tăng tốc GPU 15.9× là mức lớn nhất vì model nhỏ, graph nhiều op nhỏ.
5. **Tiền xử lý NumPy**: legacy 0.56 ms, nhưng biến thể **linear lên 14.8 ms** vì
   `resample_xy` lặp Python theo từng khớp (25 khớp × 64 vị trí). Đây là ứng viên
   sửa đầu tiên nếu dùng nội suy linear ở tần suất cao.
6. **Decode**: 1.5–2.2 ms/frame, không tăng tốc được bằng GPU trong kiến trúc hiện tại.

**Ảnh hưởng của số luồng CPU** (median ms, cùng đầu vào)

| Luồng | YOLOX detect | RTMPose/crop | Fall ONNX | Tăng tốc detect | Tăng tốc classify |
|---:|---:|---:|---:|---:|---:|
| 1 | 396.3 | 72.5 | 300.4 | 1.00× | 1.00× |
| 2 | 223.9 | 40.3 | 194.5 | 1.77× | 1.54× |
| 4 | 121.4 | 26.1 | 149.3 | 3.26× | 2.01× |
| 8 | 76.8 | 17.8 | 126.7 | 5.16× | 2.37× |
| 16 | 76.4 | 14.1 | 132.2 | 5.19× | 2.27× |

Detector scale tốt đến 8 luồng rồi đi ngang (12 nhân vật); classifier đạt hiệu quả
ở 8 luồng. `--threads 8` tiết kiệm ~52.9 ms/frame theo micro-benchmark
(detect 121.4→76.8 ms, pose 26.1→17.8 ms), tức CPU end-to-end khoảng **1.35×**
(ước lượng, chưa đo lại toàn bộ pipeline), vẫn kém GPU ~2.4×.

## 7. Khởi động lạnh (tiến trình mới)

| Bước | CPU | CUDA |
|---|---:|---:|
| Tạo 3 session ONNX | 460 ms | 620 ms |
| YOLOX — lần đầu / lần sau | 125 / 125 ms | 633 / 20 ms |
| RTMPose — lần đầu / lần sau | 27 / 25 ms | 180 / 6 ms |
| Fall ONNX — lần đầu / lần sau | 187 / 159 ms | 412 / 10 ms |
| **Tới kết quả đầu tiên** | **0.41 s** | **1.28 s** |

Lần chạy CUDA đầu tiên trên máy còn tốn thêm thời gian khởi tạo driver/GPU
(quan sát ~60 s ở lần chạy đầu tiên của phiên benchmark), các lần sau nằm trong
bảng trên.

## 8. Kết luận và đề xuất

1. **Chạy thật trên GPU, tắt overlay**: 33.2 FPS (1.10× realtime) ở 1080p/30 FPS;
   kèm overlay thì 18.4 FPS. `--no-render` đã được thêm vào `infer_video.py`:
   vẫn ghi `windows.csv`, `frames.jsonl` (rỗng), `poses.npz`, `summary.json`,
   không ghi `annotated.mp4`.
2. **Giảm tải YOLOX**: đây là 38% (GPU) / 67% (CPU). Các hướng theo thứ tự rẻ:
   chạy detector ở 640×640 cố định và bỏ letterbox lặp lại (đã có), dùng bản
   nano/tiny, hoặc chỉ chạy detector mỗi N frame và track bbox giữa các frame
   (đã có tracker tâm hông, cần thêm đo ảnh hưởng đến recall).
3. **Batching RTMPose**: đo được −18% cho 2 người; ONNX đã hỗ trợ batch động.
4. **Vectorize `resample_xy`** nếu chuyển sang `--temporal-method linear` ở tần suất
   cao (14.8 ms/cửa sổ hiện tại, thuần vòng lặp Python).
5. **CPU**: đặt `--threads 8`; vẫn cần GPU cho realtime.
6. **Bộ nhớ**: RSS đỉnh ~1.5 GB (chủ yếu do ONNX Runtime + video buffer); không
   phải ràng buộc trên GPU 12 GB.

### Phạm vi và giới hạn

- Số đo trên 4 video 1080p của bộ `PdPdFall-ONNX`, 1–2 người/cảnh; chưa đo tải
  nhiều người (ví dụ 5–10 người/cảnh) — chi phí pose scale theo số người.
- Máy đo dùng Xeon E5 (2.5 GHz base); CPU khác sẽ cho số khác.
- Cùng một bộ cửa sổ/threshold như production (window 64, stride 8, min-frames 16,
  max-gap 8, min-coverage 0.6, threshold 0.5, detector 0.5, joint 0.3, torso 0.3).
- Độ trễ đầu vào ở dạng file; chưa đo camera stream (USB/IP), nơi decode và
  transfer mạng có thể thêm độ trễ.

### Artifact

- `outputs/benchmark/cuda/results.json`, `outputs/benchmark/cpu/results.json`:
  môi trường, thông số model (kèm SHA256), breakdown từng video, cold start,
  thread scaling.
- `outputs/benchmark/<device>/micro.json`: độ trễ từng thành phần (median/mean/p95).
- `outputs/benchmark/<device>/<video>/annotated.mp4`: video overlay sinh ra trong
  lúc đo (giống hệt output production).
