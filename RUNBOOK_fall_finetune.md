# NeuroPath fall finetune — runbook (NTU RGB+D 120, binary falling / not-falling)

Source model: NeuroPath EGGCN pretrained NTU60 xsub checkpoint
`pre_trained/weights-54-16902.pt` (included in this zip).

## 1. Environment (uv)
    uv venv && uv sync            # dependencies in pyproject.toml + uv.lock
    # (torch 2.14.1+cu130; needs CUDA GPUs)

## 2. Data
Preferred: copy the prebuilt arrays (already normalised MS-G3D style, T=150):

    data/ntu120_fall_v2/x_train.npy   (~2.64 GiB — sync from cloud)
    data/ntu120_fall_v2/x_test.npy    (~2.13 GiB — sync from cloud)
    data/ntu120_fall_v2/y_train.npy, y_test.npy, manifest.csv, stats.json  (in zip)

Or regenerate from raw NTU RGB+D 120 `.skeleton` files (25 GB, 114,480 files):

    uv run python data_gen/prepare_ntu120_fall_v2.py --data_dir skeleton_data \
        --out_dir data/ntu120_fall_v2 --workers 4
    # label: A043 = falling (1); 113 other actions = 0
    # official NTU120 cross-subject split; excludes 535 missing-skeleton files

## 3. Finetune (single-run: 24 epochs, ~40 min on 2x T4)

    uv run python main.py --config config/fall-detection/train_fall_v2.yaml

- Loads the pretrained checkpoint and removes fc_1/fc_2 (60-class heads),
  trains a new binary head; small LR for backbone + x10 for the new head.
- Keeps all 671 train falls + 8x non-fall negatives per epoch (re-drawn each
  epoch). Best model selected by fall F1 on an 8:1 test subset.
- Work dir resolves to `work_dir/fall/xsub/finetune_eggcn_p` (parser appends
  the `_p` suffix for this config). Best weights:
  `work_dir/fall/xsub/finetune_eggcn_p/best/weights-*.pt`

## 4. Final evaluation (full 50,919 test clips, natural prevalence 0.54%)

    uv run python eval_fall.py --config config/fall-detection/train_fall_v2.yaml \
        --weights work_dir/fall/xsub/finetune_eggcn_p/best/weights-<best>.pt

Reports accuracy / balanced accuracy / fall precision-recall-F1 at 0.5,
confusion matrix, ROC-AUC, PR-AUC, threshold sweep and best-F1 threshold.
Saves scores to `fall_test_scores_finetuned.npy`.

## 3-epoch reference (already run on this machine)
| epoch | fall F1 (8:1 subset) | P | R |
|---|---|---|---|
| 1 | 86.97% | 78.76% | 97.09% |
| 2 | 89.37% | 82.26% | 97.82% |
| 3 | 91.89% | 97.94% | 86.55% |

Single-instance guard: `work_dir/.train.lock` blocks duplicate concurrent
trainings; delete the lock file to run a new training deliberately.
