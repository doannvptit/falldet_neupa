#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a fall-detection fine-tuned model on the FULL (realistic) test set.

- Loads feeder/model args from the training config YAML (feeder_fall, eggcn23).
- Runs on the full NTU120 cross-subject test partition with its natural class
  distribution (fall=0.54%) and reports accuracy, balanced accuracy,
  precision/recall/F1 (fall class), confusion matrix, ROC-AUC, PR-AUC, and a
  threshold sweep (with the best-F1 threshold).

NOTE: there is intentionally *no* zero-shot "falling" baseline: the pretrained
checkpoint is a 60-class NTU60 model and falling (A043) only exists in NTU120,
so no output head of the pretrained model is semantically "falling".
NTU60 class 42 ("wipe face") is NOT falling.

Usage:
  uv run python eval_fall.py [--config config/fall-detection/train_fall_v2.yaml]
                             [--weights work_dir/.../best/weights-XX.pt]
                             [--batch 128] [--seed 1024]
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from feeders.feeder_fall import Feeder
from utils import import_class, init_seed


def compute_work_dir(cfg):
    # NOTE: parser.check_args uses hasattr(dict, 'num_groups') (always False) so
    # the effective work_dir is exactly what the config declares.
    return cfg['work_dir']


def find_best_weights(work_dir):
    best = os.path.join(work_dir, 'best')
    cands = sorted([f for f in os.listdir(best)
                    if f.startswith('weights-') and f.endswith('.pt')])
    if not cands:
        sys.exit(f'no best weights found under {best}')
    return os.path.join(best, cands[-1])


def build_loader(feeder_args, batch_size, num_workers=2):
    args = dict(feeder_args)
    args['subsample_ratio'] = 0        # full, realistic class distribution
    args['random_rot'] = False
    args['debug'] = False
    if 'bone' not in args:
        args['bone'] = False
    if 'vel' not in args:
        args['vel'] = False
    if 'multi_input' not in args:
        args['multi_input'] = False
    ds = Feeder(**args)
    print(f'{args["split"]:>5}: {len(ds)} samples | falls = {int(ds.label.sum())}')
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      drop_last=False)


@torch.no_grad()
def collect_scores(model, loader, output_device, batch_split=64):
    """Return y (N,), p_fall (N,) where p_fall = softmax[:, 1]."""
    ys, ps = [], []
    model.eval()
    for data, label, _ in tqdm(loader, desc="Full test", dynamic_ncols=True):
        data = data.float().to(output_device, non_blocking=True)
        bsz = data.size(0)
        if bsz <= batch_split:
            o1, o2, o = model(data)
            prob = torch.softmax(o, dim=1)[:, 1]
        else:
            prob = []
            for s in range(0, bsz, batch_split):
                o1, o2, o = model(data[s:s + batch_split])
                prob.append(torch.softmax(o, dim=1)[:, 1])
            prob = torch.cat(prob)
        ys.append(label.numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def report(y, p, title):
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).mean()
    bal_acc = balanced_accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    auc_roc = roc_auc_score(y, p)
    auc_pr = average_precision_score(y, p)
    print(f'\n===== {title} (decision threshold = 0.50) =====')
    print(f'  accuracy {acc * 100:.2f}% | balanced-accuracy {bal_acc * 100:.2f}% | '
          f'ROC-AUC {auc_roc:.4f} | PR-AUC {auc_pr:.4f}')
    print(f'  fall: precision {prec[0] * 100:.2f}% | recall {rec[0] * 100:.2f}% | '
          f'F1 {f1[0] * 100:.2f}%')
    print(f'  confusion [tn fp / fn tp] = {cm.tolist()}')
    metrics = dict(acc=float(acc), bal_acc=float(bal_acc),
                   auc_roc=float(auc_roc), auc_pr=float(auc_pr),
                   precision=float(prec[0]), recall=float(rec[0]),
                   f1=float(f1[0]), cm=cm.tolist(), threshold=0.5)

    print(f'----- {title}: threshold sweep -----')
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pr = (p >= th).astype(int)
        pr_, rc_, f1_, _ = precision_recall_fscore_support(
            y, pr, labels=[1], zero_division=0)
        print(f'  th={th:.2f} | P {pr_[0] * 100:5.2f}% | R {rc_[0] * 100:5.2f}% | '
              f'F1 {f1_[0] * 100:5.2f}%')

    pr_curve, rc_curve, ths = precision_recall_curve(y, p)
    f1s = 2 * pr_curve[:-1] * rc_curve[:-1] / np.maximum(
        pr_curve[:-1] + rc_curve[:-1], 1e-9)
    i = int(f1s.argmax())
    best_pred = (p >= ths[i]).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, best_pred, labels=[1], zero_division=0)
    print(f'  best-F1 threshold = {ths[i]:.3f} -> F1 {f1[0] * 100:.2f}% '
          f'| P {prec[0] * 100:.2f}% | R {rec[0] * 100:.2f}%')
    metrics['th_star'] = float(ths[i])
    # This test-set optimum is descriptive; choose deployment thresholds on validation data.
    metrics['best_threshold_on_test'] = dict(
        threshold=float(ths[i]), precision=float(prec[0]), recall=float(rec[0]),
        f1=float(f1[0]), cm=confusion_matrix(y, best_pred, labels=[0, 1]).tolist())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/fall-detection/train_fall_v2.yaml')
    ap.add_argument('--weights', default=None,
                    help='finetuned weights (default: newest best in work_dir)')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=1024)
    ap.add_argument('--output-dir', default='.',
                    help='directory for test scores and JSON metrics')
    args = ap.parse_args()
    init_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loader = build_loader(cfg['test_feeder_args'], args.batch, cfg.get('num_worker', 2))
    out_device = device

    # ---------------- fine-tuned binary model ----------------
    Model = import_class(cfg['model'])
    work_dir = compute_work_dir(cfg)
    finetuned_weights = args.weights or find_best_weights(work_dir)
    model = Model(**cfg['model_args']).to(device)
    weights = torch.load(finetuned_weights, map_location=device)
    weights = OrderedDict([[k.split('module.')[-1], w] for k, w in weights.items()])
    state = model.state_dict()
    state.update(weights)
    model.load_state_dict(state)
    model.eval()
    print(f'finetuned weights: {finetuned_weights}')

    y, p = collect_scores(model, loader, out_device, args.batch)
    m = report(y, p, 'FINETUNED (binary head) on full test set')
    view_metrics = {}
    if cfg['test_feeder_args'].get('image2d_multiview', False):
        views = loader.dataset.eval_views
        for index, view in enumerate(views):
            view_metrics[view] = report(y[index::len(views)], p[index::len(views)],
                                        f'FULL TEST / synthetic camera {view}')
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'fall_test_scores_finetuned.npy'),
            np.stack([y, p]), allow_pickle=False)
    with open(os.path.join(args.output_dir, 'fall_test_metrics.json'), 'w') as f:
        json.dump(dict(config=args.config, weights=finetuned_weights,
                       samples=len(y), falls=int(y.sum()), metrics=m,
                       synthetic_view_metrics=view_metrics), f, indent=2)
    print('\nmetrics:', m)


if __name__ == '__main__':
    main()
