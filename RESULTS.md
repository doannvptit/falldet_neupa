# YOLOX + RTMPose26 inference verification

All three ONNX models ran on RTX 3060 via CUDAExecutionProvider; CPU also verified.
17 runtime tests passed. Z=0, NTU mapping, causal overlays and H.264 frame counts verified.

| Video | Frames | Windows | FALL windows | First alert (s) |
|---|---:|---:|---:|---:|
| 20240912_101331 | 56 | 6 | 2 | 1.567 |
| 20240912_101427 | 57 | 7 | 5 | 1.033 |
| demo | 282 | 52 | 4 | 7.167 |
| fall_0002 | 663 | 59 | 0 | — |

fall_0002 has no alert at threshold 0.5; the prior MediaPipe adapter had alerts.
Changing pose estimator changes the skeleton distribution. No ground truth was supplied;
these checks establish runtime correctness, not detection accuracy.
