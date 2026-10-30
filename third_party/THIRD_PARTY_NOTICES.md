# RTMPose / YOLOX preprocessing

`fall_onnx/pose_helpers.py` and preprocessing / decoding methods in
`fall_onnx/pose.py` derive from rtmlib version 0.0.16 (Apache License 2.0).
Source: https://github.com/Tau-J/rtmlib
License: RTMLIB_LICENSE in this directory.

Modifications: local model files only; explicit ONNX session providers and CPU
thread limits; enforce Halpe26 shape; avoid full-image pose inference when no
person is detected; configurable embedded-NMS confidence threshold.
The pinned exported YOLOX model already contains NMS.

Model provenance and SHA256: ../models/pose_manifest.json.
Keypoint layout: https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/halpe26.py
