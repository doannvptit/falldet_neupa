# Đánh giá video 8 FPS bằng nội suy skeleton XY

Đã lấy mẫu RGB từ 4 video xuống 8 FPS rồi chạy lại YOLOX-s/RTMPose26 trên RTX 3060.
Model fall ONNX, adapter, ngưỡng 0.5 và độ phân giải giữ nguyên. Không train lại.
Các frame RGB được lấy tại floor(n × FPS nguồn / 8), mã hóa RGB lossless và kiểm tra hash pixel.
Tốc độ phát giữ theo thời gian thực; sai khác độ dài do làm tròn nhỏ hơn 0.125 giây.

## Thời điểm cảnh báo đầu tiên

| Video | 30 FPS cũ | 8 FPS, linear 2s | 8 FPS, linear 4s | 8 FPS, linear 8s | Nearest 4s |
|---|---:|---:|---:|---:|---:|
| 20240912_101331 | 1.567s | 1.625s | 1.625s | 1.625s | — |
| 20240912_101427 | 1.033s | 1.125s | 1.125s | 1.125s | — |
| demo | 7.167s | 6.875s | 3.625s | 3.625s | 3.875s |
| fall_0002 | — | — | — | — | — |

Các mốc trên là thời điểm có đủ dữ liệu để phát cảnh báo; không phải độ trễ tính từ lúc bắt đầu ngã.
Chưa có nhãn khoảng ngã nên không tính accuracy/precision/recall hay kết luận cảnh báo sớm hơn là tốt hơn.

## Cấu hình đã thử

- `legacy_64frames`: giữ window=64, stride=8, min-observed=16; ở 8 FPS là tối đa 8 giây, cảnh báo mỗi 1 giây.
- `linear_2s`: window=16 frame nguồn → nội suy 64 pose; stride=2 (0.25 giây), min-observed=8.
- `linear_4s`: window=32 → 64 pose; stride/min-observed như trên.
- `linear_8s`: window=64 → 64 pose, chủ yếu nội suy khớp/frame thiếu ngắn.
- `nearest_4s`: cùng window/stride/warmup với linear_4s, thay nội suy tuyến tính bằng nearest.
- `linear_8s_full`: chờ đủ 64 vị trí frame nguồn trong một segment mới đánh giá; vẫn yêu cầu coverage>=60%.

Cấu hình partial bắt đầu sau ít nhất 8 frame pose quan sát thật và coverage>=60%.
Window 4/8 giây là độ dài tối đa, không phải luôn chờ đủ 4/8 giây lúc khởi động.
Hai clip ngắn dưới 2 giây nên các window partial 2/4/8 giây dùng cùng lịch sử và có cùng kết quả.
Không nối dài clip, lặp cuối clip hoặc tính frame tổng hợp là quan sát thật.

## Khi chờ đủ cửa sổ 8 giây

| Video | Frame 8 FPS | Cửa sổ đủ 64 frame | FALL windows |
|---|---:|---:|---:|
| 20240912_101331 | 15 | 0 | 0 |
| 20240912_101427 | 16 | 0 | 0 |
| demo | 76 | 0 | 0 |
| fall_0002 | 177 | 33 | 0 |

Không có cửa sổ đầy đủ nghĩa là chưa đủ lịch sử usable liên tục, không phải dự đoán NON-FALL.
demo có 76 frame toàn video nhưng các segment theo người không đủ 64 frame liên tục đạt điều kiện.

## Nhận xét

- Trên bộ thử này, linear 2s giữ cảnh báo ở hai clip ngắn, chậm hơn mốc cảnh báo 30 FPS khoảng 58ms và 92ms.
- Nearest 4s không vượt ngưỡng ở hai clip ngắn; linear 4s có cảnh báo. Đây là đối chứng cùng warmup/window.
- demo: linear 2s cảnh báo 6.875s, gần mốc cũ 7.167s; linear 4/8s báo từ 3.625s. Cần nhãn để đánh giá báo sớm/sai.
- fall_0002 vẫn không vượt ngưỡng ở mọi cấu hình; nội suy không khắc phục được hạn chế estimator/adapter/model.
- Cấu hình thử nghiệm phù hợp để tiếp tục đánh giá: linear 2s, 16 frame nguồn → 64 pose, stride=2, max-gap=2.
- Chờ đủ 8 giây tạo thời gian chờ lớn và bỏ qua các clip ngắn; chưa thấy lợi ích trên các video này.

## Nội suy và tính nhân quả

Nội suy tuyến tính từng khớp XY, chỉ giữa hai quan sát hợp lệ; Z=0.
Không ngoại suy ngoài hai đầu hoặc nối qua hơn 2 frame thiếu (~0.25 giây tại 8 FPS).
Pose tổng hợp chỉ được tính khi đã nhận frame cuối cửa sổ. Không phát cảnh báo ngược về thời điểm pose trung gian.
Video preview nội suy là minh họa offline, còn annotated.mp4 trong từng policy giữ đúng timeline 8 FPS.
64 pose nội suy không mang lại 64 lần quan sát độc lập và không phục hồi chuyển động đã mất giữa frame nguồn.

## Chạy lại

```bash
cd /mnt/doannv/home/lumi/luvox-project/falldet/poc_inference_onnx
.venv/bin/python scripts/evaluate_8fps.py --render
# Dùng lại cache 8 FPS, không trích xuất pose lại:
.venv/bin/python scripts/evaluate_8fps.py --reuse --render
.venv/bin/python scripts/report_8fps.py
```

Với video đầu vào đã khoảng 8 FPS:

```bash
.venv/bin/python infer_video.py --video video_8fps.mp4 --device cuda \
  --window 16 --stride 2 --min-frames 8 --max-gap 2 \
  --temporal-method linear --interpolation-gap 2 --output-dir outputs/lowfps
```

## Artifacts

- `outputs/eval_8fps/results.json`: toàn bộ score summary, chính sách và sampling manifest.
- `outputs/eval_8fps/<policy>/<video>/`: windows.csv, summary.json, annotated.mp4, frames.jsonl.
- `outputs/eval_8fps/previews/`: video so sánh pose quan sát và pose nội suy, NumPy 64 pose và tensor model.
- `outputs/eval_8fps/verification.json`: kiểm tra 24 video policy, số frame/FPS, Z=0, timeline nhân quả và cache.
- `tests.log`: 24/24 tests pass.
