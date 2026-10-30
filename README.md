# Fall detection 2D — ONNX độc lập

Source này chỉ dùng NumPy + ONNX Runtime cho bộ phân loại fall/non-fall. Không cần
PyTorch, checkpoint `.pt`, dataset, `PYTHONPATH` hay import bất cứ module nào từ `poc`.
Có thể sao chép nguyên thư mục này sang máy khác để chạy. Pipeline video mặc định hiện là
**YOLOX-s → RTMPose-m Halpe26 → NTU25 → fall ONNX**, thay thế MediaPipe.

## Mô hình

- `models/fall_image2d.onnx`: EGGCN binary, checkpoint multiview 2D epoch 2.
- Input `skeleton`: float32 **[1,3,64,25,2]** = batch, XYZ, thời gian, khớp NTU25, người.
  Giữ 3 kênh theo kiến trúc pretrained; **Z luôn bằng 0**.
- Output `logits`: float32 **[1,2]**, thứ tự `NON-FALL`, `FALL`.
  Áp dụng softmax rồi so xác suất lớp 1 với threshold mặc định 0.5.
- Export FP32, opset 17, shape cố định. Giữ cả hai nhánh joint/motion và fused head.
- Mỗi track được phân loại riêng, slot người thứ hai bằng 0; không ước lượng 3D.
- `models/fall_image2d.json`: SHA256 checkpoint/ONNX và kết quả kiểm tra export.
- `models/yolox_s.onnx`: phát hiện người, input `[1,3,640,640]`.
- `models/rtmpose_m_halpe26.onnx`: pose 26 khớp, input `[1,3,256,192]`.
- `models/pose_manifest.json`: nguồn tải OpenMMLab, SHA256 và kích thước model.
- `fall_onnx/halpe26.py`: adapter có version, xử lý confidence và khớp thiếu.
- Không cần cài `mediapipe`, `rtmlib`, MMPose hoặc PyTorch. Các hàm crop/decode
  tham chiếu từ rtmlib được đóng gói trong source, kèm giấy phép Apache-2.0
  tại `third_party/`. Asset MediaPipe cũ được giữ để lưu trữ, không được nạp.

## Cài đặt

Python 3.12 đã được kiểm chứng. Trên máy hiện tại `.venv` đã cấu hình **GPU CUDA 13**,
không có torch/mediapipe/rtmlib. Các lệnh cài dưới đây dành cho máy mới.
Có thể dùng `--device cpu` với bản ONNX Runtime GPU hiện tại.

```bash
cd /mnt/doannv/home/lumi/luvox-project/falldet/poc_inference_onnx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Nếu cần đọc video:
python -m pip install -r requirements-video.txt
```

Video cần `ffmpeg` có encoder `libx264` trong PATH. Nếu dùng uv:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-video.txt
```

`requirements-lock.txt` ghi phiên bản môi trường GPU đã chạy kiểm chứng.

Cài GPU trong môi trường riêng (driver phải hỗ trợ CUDA 13):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-gpu.txt
```

Nếu chuyển từ môi trường CPU đã cài: gỡ `onnxruntime` trước rồi cài
`requirements-gpu.txt`. Không cài đồng thời `onnxruntime` và `onnxruntime-gpu`.
Các thư viện CUDA/cuDNN cần thiết được khai báo trong requirements; runtime nạp
qua `onnxruntime.preload_dlls()`, không tham chiếu thư viện từ source training.

Model đã tải sẵn. Trên máy mới chưa có model, tải một lần bằng:

```bash
.venv/bin/python scripts/download_pose_models.py
```

Runtime chỉ đọc model local, không tự tải mạng. Downloader lưu URL và SHA256;
chạy lại sẽ kiểm tra hash, chỉ tải đè khi truyền `--force`.

## Chạy với skeleton 2D

```bash
.venv/bin/python inference.py --input examples/pose_xy.npy
.venv/bin/python inference.py --input examples/model_input.npy --preprocessed
```

Đầu vào thường là `.npy` **[T,25,2]**, tọa độ **pixel**: X sang phải, Y xuống dưới.
Nếu pose estimator trả tọa độ chuẩn hóa, nhân X với chiều rộng và Y với chiều cao ảnh.
Thứ tự khớp phải đúng NTU25, không đưa trực tiếp MediaPipe33 hoặc COCO17 vào classifier.
Chấp nhận [T,25,3] nhưng bỏ qua hoàn toàn Z, kể cả NaN/inf tại Z.
Frame thiếu pose biểu diễn bằng toàn 0 hoặc XY NaN. Những tọa độ 0 hợp lệ vẫn được
giữ nếu cả frame không rỗng. Dùng cửa sổ 64 frame cho dữ liệu dài; API cắt tối đa
150 frame sau khi bỏ phần đầu/cuối thiếu pose, rồi nội suy về 64 như pipeline cũ.

Tiền xử lý NumPy tái hiện pipeline đã train: hold-fill khoảng trống nội bộ,
spine center, float16 round-trip, crop giữa 95%, nội suy half-pixel,
chuẩn hóa median RMS và Z=0. `--preprocessed` bỏ qua toàn bộ bước này;
tensor phải đã được chuẩn hóa đúng, không chỉ reshape XY.

API Python:

```python
import numpy as np
from fall_onnx.classifier import BinaryClassifier

model = BinaryClassifier()  # CPUExecutionProvider
poses = np.load('examples/pose_xy.npy')
result = model.predict(poses, threshold=0.5)
print(result)  # p_fall, predicted_label, label, logits
```

## Chạy với video

```bash
.venv/bin/python infer_video.py \
  --video ../PdPdFall-ONNX/videos/20240912_101331.mp4 --device cuda

.venv/bin/python infer_video.py \
  --video-dir ../PdPdFall-ONNX/videos \
  --output-dir outputs/rtmpose26_gpu --device cuda
```

Pipeline: **YOLOX-s** detector → crop từng người → **RTMPose-m Halpe26** →
ánh xạ sang NTU25 → tracking → cửa sổ nhân quả → ONNX → overlay H.264.
Tất cả đều chạy bằng ONNX Runtime. `--device cuda` áp dụng cho cả 3 model;
`--pose-device cpu` có thể chọn riêng detector/pose chạy CPU. Mặc định CLI là CPU.
Nếu yêu cầu CUDA nhưng khởi tạo CUDA provider thất bại, script báo lỗi;
không tự chuyển toàn bộ pipeline sang CPU.

Adapter Halpe26 (index NTU trong code bắt đầu từ 0):

| Nhóm | Cách ánh xạ |
|---|---|
| Vai/khuỷu/cổ tay/hông/gối/cổ chân | 12 khớp tương ứng trực tiếp |
| SpineBase, Head, Neck | Hip trung tâm, Head, Neck của Halpe; có fallback từ hông/tai/vai |
| SpineShoulder | Trung điểm 2 vai |
| SpineMid | Trung điểm SpineBase và SpineShoulder |
| Foot trái/phải | Trung bình big/small toe đang có confidence đạt ngưỡng |
| Hand, HandTip, Thumb | Proxy theo hướng khuỷu → cổ tay, hệ số kéo dài .15/.30/.10; confidence giảm một nửa |

Proxy bàn tay không phải khớp được quan sát thực tế. Khớp có nguồn không đủ
confidence được đặt 0; không dùng tọa độ pose confidence thấp như quan sát hợp lệ.
Confidence mặc định: detector=0.5, torso mean=0.3, joint source=0.3.
Frame không có spine center hợp lệ bị loại. Nếu YOLOX không phát hiện người,
không gọi pose trên toàn ảnh. Tracking dùng tâm hông theo tọa độ ảnh chuẩn hóa;
RTMPose và classifier luôn dùng XY pixel, Z=0.

Mặc định window=64, stride=8, min-frames=16, max-gap=8, min-coverage=0.6.
Khoảng mất track dài được tách segment; đoạn ít pose chưa đủ điều kiện sẽ không
được gán nhãn NON-FALL. Overlay hiển thị WARMUP/NO POSE và chỉ cập nhật khi
đã nhận đủ frame của cửa sổ, có xử lý cửa sổ cuối clip.

Mỗi video sinh thư mục `outputs/rtmpose26/<tên video>/` mặc định (hoặc `--output-dir`):

- `annotated.mp4`: video có skeleton, track ID, nhãn và xác suất.
- `windows.csv`: kết quả từng cửa sổ, frame/time bắt đầu và kết thúc.
- `frames.jsonl`: trạng thái hiển thị từng frame.
- `poses.npz`, `extraction.json`: cache raw Halpe26 XY/score/bbox, NTU25 và thông tin nguồn.
- `summary.json`: tổng hợp lần chạy.

Dùng `--reuse-poses` để chạy lại classifier từ cache trong cùng output-dir.
Nguồn video, SHA256 detector/pose, version adapter và các tham số extraction phải
khớp cache. Cache MediaPipe cũ bị từ chối. Output mới mặc định nằm trong nhánh
`outputs/rtmpose26/`, tách khỏi kết quả MediaPipe trước đây. Có thể điều chỉnh
`--threshold`, `--stride`, `--threads`; `--max-frames` dùng để chạy thử một phần.
`--no-render` bỏ bước ghi video overlay (tiết kiệm ~31% thời gian, chỉ ghi
`windows.csv`, `poses.npz`, `summary.json`).


## Video khoảng 8 FPS

Đã thử 4 video ở 8 FPS với nội suy skeleton XY, Z=0 trên GPU. Kết quả,
so sánh cửa sổ 2/4/8 giây và các giới hạn nằm trong [EVAL_8FPS.md](EVAL_8FPS.md).
Video minh họa nằm trong `outputs/eval_8fps/previews/`; RGB không được nội suy.

Với video đầu vào đã khoảng 8 FPS, cấu hình thử nghiệm dùng tối đa 16 frame
nguồn (2 giây), nội suy thành 64 pose cho model:

```bash
.venv/bin/python infer_video.py --video video_8fps.mp4 --device cuda \
  --window 16 --stride 2 --min-frames 8 --max-gap 2 \
  --temporal-method linear --interpolation-gap 2 --output-dir outputs/lowfps
```

Cảnh báo được cập nhật mỗi 2 frame nguồn (0.25 giây). Cửa sổ ban đầu được
đánh giá khi có ít nhất 8 quan sát thật và coverage đạt 60%. Dùng `--window 32`
cho lịch sử tối đa 4 giây hoặc `--window 64` cho 8 giây; thêm
`--require-full-window` nếu cần chờ đủ chiều dài cửa sổ trước khi đánh giá.
Các tùy chọn này không tự lấy mẫu video FPS cao xuống 8 FPS; script
`scripts/evaluate_8fps.py --render` thực hiện bước lấy mẫu khi đánh giá.

## Kiểm chứng

```bash
.venv/bin/python -m unittest discover -s tests -v
```

- Model fall giữ nguyên: ONNX checker và so logits PyTorch/ONNX trên 7 input.
- 24 tests: tiền xử lý, nội suy XY và cửa sổ thời gian, score tham chiếu, mapping trái/phải, spine/foot/hand proxy,
  confidence/missing joints, không pose khi detector rỗng, ngưỡng NMS,
  track không phụ thuộc thứ tự detections và cache không dùng nhầm adapter/model.
- Chạy mới toàn bộ 4 video trên CPU và RTX 3060; báo cáo tại
  `integration_validation.json`, video GPU tại `outputs/rtmpose26_gpu/`.
- `validation.json` là kiểm tra export/tiền xử lý với skeleton MediaPipe trước
  khi thay pose estimator; không phải benchmark độ chính xác RTMPose.
- Báo cáo cũ được lưu tại `reports/mediapipe_previous/`.

## Benchmark hiệu năng

Đo trên RTX 3060 và Xeon E5-2678 v3: độ trễ từng thành phần (decode, YOLOX,
RTMPose, mapping, tracking, tiền xử lý, ONNX), phân bổ thời gian end-to-end,
cold start và mở rộng số luồng CPU. Báo cáo ở [BENCHMARK.md](BENCHMARK.md).

```bash
.venv/bin/python scripts/benchmark_pipeline.py --device cuda --threads 4 --repeat 20
.venv/bin/python scripts/benchmark_pipeline.py --device cpu  --threads 4 --repeat 20
```

Tóm tắt: GPU 18.4 FPS có overlay, **33.2 FPS khi bỏ overlay** (1.10× realtime
1080p/30 FPS); CPU 4.8 FPS. Nút thắt cổ chai là YOLOX-s (38% GPU / 67% CPU)
và bước encode overlay (31% GPU).

Hai video ngắn có cảnh báo FALL; với pipeline mới `fall_0002.mp4` chưa có cửa sổ
vượt ngưỡng 0.5. Việc thay estimator/adapter làm thay đổi phân bố skeleton;
chưa có ground truth để tính accuracy/precision/recall. Cần đánh giá dữ liệu camera
thực và có thể fine-tune với skeleton từ RTMPose, đặc biệt do bàn tay dùng proxy.
Không hạ threshold dựa trên một clip để coi như đã giải quyết chất lượng.

## Xuất lại từ checkpoint (chỉ dành cho môi trường training)

File ONNX đã được đóng gói sẵn. Khi cần xuất lại, chạy trong **poc**:

```bash
cd ../poc
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/export_fall_image2d_onnx.py
# Đối chiếu pipeline với mã training và các cache hiện có:
.venv/bin/python scripts/verify_fall_onnx_bundle.py
```

Exporter cần `torch`, `onnx`, `onnxruntime` trong môi trường training.
Các script export/đối chiếu chỉ dùng lúc phát triển, không thuộc dependency runtime.
