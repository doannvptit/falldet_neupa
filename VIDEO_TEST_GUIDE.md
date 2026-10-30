# Video mô phỏng và test video RGB

Pipeline MediaPipe dùng checkpoint binary epoch 2 đã được triển khai tại
`infer_fall_video.py`. Xem [hướng dẫn cài và chạy](MEDIAPIPE_FALL.md).
Mặc định mới dùng [pose ảnh 2D đa góc nhìn](FINETUNE_MULTIVIEW_2D.md), không
cần world pose 3D ở classifier.
Phần so sánh pipeline ONNX bên dưới giải thích khác biệt với repo cũ.

## Video đã tạo

`outputs/fall_sample/fall_skeleton.mp4` dựng lại mẫu test đầu tiên có nhãn fall:
`S001C001P003R001A043.skeleton`, test index 42, 71 frame.

- Bên trái: chuyển động 3D gốc, giữ vị trí trong hệ tọa độ camera.
- Bên phải: dữ liệu thực sự từ `x_test.npy`, đã center/rotate theo preprocessing.
  Các trục của dữ liệu chuẩn hóa được giữ nguyên khi vẽ; vị trí và hướng hiển
  thị khác bên trái do phép chuẩn hóa.
- Đây là video vẽ skeleton, không phải video RGB khôi phục từ skeleton.
- Nhãn fall là nhãn của cả clip. Kết quả dự đoán trên video cũng là kết quả
  toàn clip, không phải bộ phát hiện thời điểm bắt đầu ngã.
- Video phát ở 15 fps để dễ quan sát. Đây là tốc độ hiển thị được chọn,
  không suy ra timestamp gốc từ file skeleton.

Artifact cùng thư mục: `fall_timeline.png`, `metadata.json`,
`sample_normalized.npy` và `model_input.npy`.
`model_input.npy` có shape `[1,3,64,25,2]` float32, đã qua feeder giống khi test.

Chạy lại từ thư mục `poc`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python render_fall_sample.py
```

## Thư mục video RGB hiện có

`../PdPdFall-ONNX/videos`:

| Video | Độ phân giải | Thời lượng | Frame |
|---|---|---|---|
| `20240912_101331.mp4` | 1920×1080 | 1,90 s | 56 |
| `20240912_101427.mp4` | 1920×1080 | 1,90 s | 57 |
| `demo.mp4` | 1920×1080 | 9,40 s | 282 |
| `fall_0002.mp4` | 1920×1080 | 22,11 s | 663 |

Các video xấp xỉ 30 fps. Thông tin lấy bằng `ffprobe`.

## Chạy pipeline ONNX hiện có

Repo bên cạnh đã có detector PP-YOLOE, pose HRNet, tracker và STGCN,
cùng các file model. Môi trường `.venv` của repo đó đã import được OpenCV
5.0.0 và ONNX Runtime 1.17.3. Provider hiện thấy là Azure và CPU; chưa có CUDA.

Lệnh dưới đây test **STGCN hiện có**, không dùng checkpoint binary vừa finetune.
Lệnh chưa được chạy xử lý các video trong phiên này:

```bash
ONNX_ROOT=/mnt/doannv/home/lumi/luvox-project/falldet/PdPdFall-ONNX
POC_ROOT=/mnt/doannv/home/lumi/luvox-project/falldet/poc

"$ONNX_ROOT/.venv/bin/python" "$ONNX_ROOT/run.py" \
  --video "$ONNX_ROOT/videos/demo.mp4" \
  --out "$POC_ROOT/outputs/rgb_test/demo.mp4" \
  --ensemble stgcn_only --no_ctrgcn --no_neuropath --no_gpu
```

Đổi `--video` và `--out` để chạy video khác. Bỏ `--no_gpu` chỉ có tác dụng dùng
GPU khi môi trường ONNX Runtime thực sự có CUDAExecutionProvider và truy cập
được thiết bị; hiện tại wrapper tự fallback CPU. FFmpeg đã có sẵn để xuất H.264.

## Yêu cầu tương thích model binary và pipeline ONNX cũ

Luồng xử lý cần có:

`RGB → phát hiện người → tracking → pose theo thời gian → adapter/preprocessing → binary NeuroPath → overlay/kết quả`

Detector, tracker và pose 2D đã có trong repo bên cạnh. Phần classifier và
adapter cần được sửa để tương thích checkpoint finetune:

| Thành phần | Pipeline NeuroPath ONNX hiện có | Model binary vừa finetune |
|---|---|---|
| Pose nguồn | COCO 17 khớp 2D | Được train bằng NTU 25 khớp 3D |
| Đầu vào classifier | `[B,2,3,64,25,2]`, joint + motion bên ngoài | `[B,3,64,25,2]`, motion được tính bên trong |
| Số lớp | 60, đọc fall ở index 42 | 2, `0=non-fall`, `1=fall` |
| Chuẩn hóa adapter | Tạo Z=0, map khớp heuristic, trừ root và chia 256 | Pipeline train center spine theo frame và rotate, crop/resize về 64 frame |
| Convention overlay | `0=fall`, `1=normal` | Cần chuyển từ nhãn classifier sang convention overlay |

1. **Trọng số đúng:** dùng `work_dir/fall/verify_3epoch/train/best/weights-2-17090.pt`.
   Có thể infer bằng PyTorch trực tiếp. Nếu dùng ONNX Runtime thì export model
   binary này sang ONNX và kiểm tra logits/probability khớp PyTorch trên cùng
   `model_input.npy`; export output fused logits 2 lớp.
2. **Adapter dữ liệu đúng:** input float32 `[B,3,64,25,2]`, đúng thứ tự khớp,
   trục tọa độ, xử lý frame mất và người thứ hai, cùng preprocessing đã train.
   Không đưa thêm chiều modality hay tự tạo motion ngoài model cho cấu hình này.
3. **Giải quyết chênh lệch 2D/3D:** adapter hiện có chỉ map 17→25 và chèn Z=0,
   không tái tạo được skeleton Kinect 3D. Để kiểm chứng phù hợp hơn, cần pose 3D
   và mapping/chuẩn hóa tương thích; hoặc fine-tune/đánh giá lại bằng dữ liệu
   pose 2D lấy từ chính pipeline RGB. Chỉ thay model ONNX chưa giải quyết được
   chênh lệch này, và không thể mang F1 trên skeleton NTU sang video RGB.
4. **Đọc output đúng:** `p_fall = softmax(fused_logits)[...,1]`, ngưỡng ban đầu
   0,5 theo lần test skeleton. Wrapper cũ đọc index 42 sẽ không dùng được cho
   output 2 lớp. Cần đổi nhãn khi đi qua overlay vốn dùng `0=fall`.
5. **Cửa sổ thời gian và cuối video:** code hiện buffer 50 frame, classifier
   NeuroPath resize về 64. Hai clip ngắn chỉ có 56–57 frame; mất track dễ khiến
   không đủ cửa sổ. `run.py` hiện không flush phần buffer còn lại ở EOF, nên
   cần xử lý phần cuối clip khi tích hợp và kiểm tra độ trễ/cửa sổ trượt.
6. **Nhãn kiểm chứng:** ghi nhãn fall/non-fall của từng clip, tốt hơn có khoảng
   thời gian ngã và người tương ứng, để tính false positive/false negative,
   precision/recall/F1 và độ trễ. Tên file video chưa đủ để xác định ground truth.

Video skeleton đã dựng có thể test trực tiếp classifier bằng `model_input.npy`.
Không dùng việc chạy person detector/HRNet trên hình stick-figure này để đo
chất lượng pipeline phát hiện người trong video RGB.
