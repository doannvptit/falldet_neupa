"""Fit one 3->2 PCA on train poses only; do not use test/video statistics."""
import argparse
import json
from pathlib import Path

import numpy as np


def fit(path, batch_size=64):
    data = np.load(path, mmap_mode='r')
    n, total, cross = 0, np.zeros(3, np.float64), np.zeros((3, 3), np.float64)
    for start in range(0, len(data), batch_size):
        points = np.asarray(data[start:start+batch_size], np.float64).reshape(-1, 3)
        points = points[np.any(points != 0, axis=1)]
        n += len(points)
        total += points.sum(0)
        cross += points.T @ points
        if start % (batch_size * 100) == 0:
            print(f'Fit PCA: {start}/{len(data)} samples, {n} nonzero joints', flush=True)
    mean = total / n
    covariance = cross / n - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, components = values[order], vectors[:, order].T
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return mean, components, values, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='data/ntu120_fall_v2/x_train.npy', type=Path)
    ap.add_argument('--output', default='data/ntu120_fall_pca2d/projection.npz', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    mean, components, values, n = fit(args.train)
    covariance = components.T @ np.diag(values) @ components
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32), components=components[:2].astype(np.float32),
             eigenvalues=values, source=str(args.train.resolve()), points=n)
    stats = dict(train=str(args.train.resolve()), points=n, mean=mean.tolist(),
                 components=components[:2].tolist(), eigenvalues=values.tolist(),
                 explained_variance_ratio=(values / values.sum()).tolist(),
                 retained_variance=float(values[:2].sum()/values.sum()),
                 discarded_variance=float(values[2]/values.sum()),
                 reconstruction_mse_per_point=float(values[2]),
                 original_xy_retained_variance=float(np.trace(covariance[:2, :2])/values.sum()),
                 note='Pooled nonzero joints after NTU 3D normalization; PCA fit on train only. Variance is not classification accuracy.')
    args.output.with_suffix('.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
