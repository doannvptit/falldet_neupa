# Fine-tune NeuroPath: Fall Detection (A43 "fall down" vs non-fall)

Fine-tune the pre-trained NeuroPath EGGCN (`pre_trained/weights-54-16902.pt`,
trained on NTU RGB+D 60 cross-subject with a 60-class head) into a **binary
fall / not-fall** detector with the full NTU RGB+D 120 skeleton dataset in
`./skeleton_data` (114,480 raw `.skeleton` files).

## 1. Setup (uv)

```bash
uv venv .venv                                          # uv-managed environment
uv add torch numpy 'pyyaml>=6' tqdm prettytable matplotlib tensorboard scikit-learn
```

Pure uv environment (no `--system-site-packages`): the system Python's
`google` namespace package would shadow the venv protobuf and crash
`torch.utils.tensorboard`. `requirements.txt` is legacy (PyYAML 5 / apex) and
NOT needed — `processor.py` / `parser.py` / `utils.py` were patched for
PyYAML 6 + PyTorch 2.x (apex optional).

## 2. Prepare binary data (one-time, ~20 min)

```bash
uv run python data_gen/prepare_ntu120_fall_v2.py --workers 4 --out_dir data/ntu120_fall_v2
```

- **Label**: `A043 (fall down)` → `1` (falling); the other 119 actions → `0`.
- **Split**: official NTU RGB+D 120 **cross-subject** partition (53 training
  subjects, exactly as `ntu120_gendata.py` of MS-G3D / paper arXiv:1905.04757).
- Excludes the 535 official `NTU_RGBD120_samples_with_missing_skeletons.txt`
  entries.
- Skeleton pre-processing is MS-G3D faithful:
  - keep up to 2 bodies per frame (`pre_normalization`: per-frame spine-base
    centering + 2 inclination rotations of person-0's first frame),
  - head-shift each body track and hold-fill internal missing frames so the
    feeder's `valid_frame_num` is always correct,
  - `T_CAP = 150` frames (99.9% of falling clips ≤ 139 frames), float16 mmap.
- Output `data/ntu120_fall_v2/` (from the last verified run):

  | file | shape / content |
  |---|---|
  | `x_train.npy` | (63,026, 150, 2, 25, 3) float16 |
  | `x_test.npy`  | (50,919, 150, 2, 25, 3) float16 |
  | `y_train.npy` | 63,026 labels — **671 falls** |
  | `y_test.npy`  | 50,919 labels — **275 falls** |
  | `manifest.csv`, `failed_files.txt`, `stats.json` | audit trail |

## 3. Fine-tune

```bash
uv run python main.py --config config/fall-detection/train_fall.yaml --phase train
```

Key choices (see `config/fall-detection/train_fall.yaml`):

| Item | Value | Rationale |
|---|---|---|
| Head | `num_class: 2`, `ignore_weights: fc_1/fc_2` | binary head re-initialized, backbone loaded from the pre-trained checkpoint |
| LR | 0.005 backbone, `head_lr_mult: 10` (new head) | fine-tune without destroying pre-trained features |
| Balance | `subsample_ratio: 8` + `class_loss_weights: [0.25, 1.0]` | 671 falls vs 62k negatives: epoch = all falls + re-drawn 8× negatives (fresh negatives every epoch), fall-weighted CE |
| Input mode | `multi_input: False` | chosen by a zero-shot A/B probe (softmax[A43] AUC) that decides which input mode matches pre-training |
| Crop | `p_interval: [0.8, 1]`, window 64, `random_rot` | covers the tail (fall segment) of each clip |
| Selection | `save_best_criterion: fall_f1` | top-1 accuracy is meaningless under imbalance |
| Optimizer | SGD Nesterov, wd 5e-4, steps [12, 20], 24 epochs, batch 64 (2×T4) | short fine-tune schedule |

Each epoch logs the fall-class confusion matrix (+ precision / recall / **F1**)
on a seeded 1:8 test subset; the best model (max F1) is kept in
`work_dir/fall/xsub/finetune_eggcn/best/`.

## 4. Full evaluation (realistic distribution)

```bash
uv run python eval_fall.py --config config/fall-detection/train_fall.yaml --baseline
```

Scores the **entire** cross-subject test partition (50,919 sequences,
99.46% non-fall) and reports accuracy, balanced accuracy, precision / recall /
F1, confusion matrix, ROC-AUC, PR-AUC and a threshold sweep (best-F1 included).
`--baseline` also scores the original 60-class pre-trained model zero-shot
(p(fall) = softmax at A43, index 42) for comparison.

## Files added/changed

- `data_gen/prepare_ntu120_fall_v2.py` — raw skeleton → binary npy converter
  (official split, MS-G3D-faithful normalization) — **the data pipeline in use**
- `feeders/feeder_fall.py` — mmap feeder, per-epoch negative re-draw (new)
- `eval_fall.py` — full-test-set evaluator + zero-shot baseline + input-mode probe (new)
- `config/fall-detection/train_fall.yaml` — final fine-tune config (new)
- `config/fall-detection/smoke.yaml` — mini pipeline smoke config (new)
- `processor.py` — apex optional, class weights, head LR group, per-epoch
  negative re-draw hook, fall-F1 logging & best-model criterion
- `feeders/tools.py` — numpy 2.x scalar fix in `valid_crop_resize`
- `parser.py`, `utils.py`, `main.py` — PyYAML 6 compat (`Loader=SafeLoader`)
- `feeders/__init__.py`, `model/__init__.py` — explicit submodule imports
- `prepare_ntu120_fall.py` — earlier (v1) converter, kept for reference only
