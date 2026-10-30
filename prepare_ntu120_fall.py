#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare NTU RGB+D 120 skeletons for binary FALL detection (finetune A43).

- Input : raw *.skeleton files (NTU RGB+D 120, 114,480 files in ./skeleton_data)
- Label : A43 "fall down" -> 1 (falling) ; all other actions -> 0 (not falling)
- Split : cross-subject = P001..P020 train / P021..P040 test (standard NTU protocol)
- Output: data/ntu120_fall/
    x_train.npy  (N_train, 300, 2, 25, 3) float16  [frame, body, joint, xyz]
    x_test.npy   (N_test , 300, 2, 25, 3) float16
    y_train.npy / y_test.npy  (N,) uint8 binary labels
    manifest.csv, failed_files.txt

Raw parsing follows the NTU skeleton file layout used by MS-G3D/ST-GCN:
  line0: numFrame
  per frame: numBody
  per body: bodyInfoLine (6 or 10 fields), numJoint-line, numJoint x joint-lines
The joint info line has 12 floats; we keep (x, y, z).

Body handling: keep up to 2 bodies with the most valid frames, forward-fill
missing frames, center each body on its first valid frame's spine base (joint 0).
"""
import argparse
import csv
import os
import re
import time
from multiprocessing import Pool

import numpy as np

FALL_ACTION = 43
N_FRAME_MAX = 300
V, M, C = 25, 2, 3

# official NTU RGB+D cross-subject partition (MS-G3D / CTR-GCN convention)
TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}

FILE_RE = re.compile(r'^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\.skeleton$')


def list_files(data_dir):
    items = []
    for name in sorted(os.listdir(data_dir)):
        m = FILE_RE.match(name)
        if m is None:
            continue
        setup, cam, subj, repl, act = map(int, m.groups())
        items.append(dict(name=name, path=os.path.join(data_dir, name),
                          setup=setup, cam=cam, subject=subj, repl=repl, action=act))
    return items


def read_skeleton_sequence(path):
    """Parse a raw .skeleton file.

    Returns list over frames; each frame is a dict bodyID -> (25,3) float32,
    or None entry if that body-in-frame is invalid (zeros / NaN / garbage).
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    if not lines:
        return None
    try:
        num_frames = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    p, total = 1, len(lines)
    frames = []
    for _ in range(num_frames):
        if p >= total:
            break
        try:
            num_body = int(lines[p].split()[0])
        except (ValueError, IndexError):
            break
        p += 1
        bodies = {}
        for _ in range(num_body):
            if p >= total:
                break
            meta = lines[p].split()
            p += 1
            body_id = meta[0]
            if p >= total:
                break
            try:
                joint_count = int(lines[p].split()[0])
            except (ValueError, IndexError):
                break
            p += 1
            nvals = min(joint_count, V)
            joints = np.zeros((V, C), dtype=np.float32)
            ok = True
            for v in range(nvals):
                if p >= total:
                    ok = False
                    break
                vals = lines[p].split()
                p += 1
                try:
                    joints[v] = (float(vals[0]), float(vals[1]), float(vals[2]))
                except (ValueError, IndexError):
                    ok = False
                    break
            if joint_count > V:
                p += joint_count - V          # skip unexpected extra joints
            if not ok or np.isnan(joints[:nvals]).any() \
                    or (joints[:nvals] == 0).all() or (np.abs(joints[:nvals]) > 1e6).any():
                bodies[body_id] = None
            else:
                bodies[body_id] = joints
        frames.append(bodies)
    return frames


def build_sample(frames):
    """frames -> (300, 2, 25, 3) float16 sample, or None if unusable."""
    if not frames:
        return None
    body_ids, seen = [], set()
    for fr in frames:
        for b, j in fr.items():
            if j is not None and b not in seen:
                seen.add(b)
                body_ids.append(b)
    if not body_ids:
        return None

    T_raw = len(frames)
    data = np.zeros((T_raw, len(body_ids), V, C), dtype=np.float32)
    id_idx = {b: i for i, b in enumerate(body_ids)}
    for t, fr in enumerate(frames):
        for b, j in fr.items():
            if b in id_idx and j is not None:
                data[t, id_idx[b]] = j

    counts = (np.abs(data).sum(axis=(2, 3)) > 0).sum(axis=0)
    order = np.argsort(counts)[::-1]
    chosen = [int(bi) for bi in order[:M] if counts[bi] >= max(8, int(0.2 * T_raw))]
    if not chosen:
        return None

    sample = np.zeros((N_FRAME_MAX, M, V, C), dtype=np.float32)
    out_T = min(T_raw, N_FRAME_MAX)
    for k, bi in enumerate(chosen):
        seq = data[:out_T, bi].copy()
        valid = np.abs(seq).sum(axis=(1, 2)) > 0
        if valid.any():
            first = int(valid.argmax())
            last = int(valid.nonzero()[0][-1])
            jt = seq[first]
            for t in range(first + 1, last + 1):
                if not valid[t]:
                    seq[t] = jt
                else:
                    jt = seq[t]
            seq -= seq[first, 0]                      # center: first frame spine-base
            seq[:first] = 0.0
            seq[last + 1:] = 0.0
            sample[:last + 1, k] = seq[:last + 1]

    if sample.shape[1] != M or np.isnan(sample).any():
        return None
    return sample.astype(np.float16)


def process_chunk(args_tuple):
    chunk, data_dir = args_tuple
    results = []
    failures = []
    for (idx, fname, label, split) in chunk:
        x = None
        try:
            frames = read_skeleton_sequence(os.path.join(data_dir, fname))
            x = build_sample(frames)
            why = 'empty / too few valid frames'
        except Exception as exc:
            why = repr(exc)[:150]
        if x is None:
            failures.append((fname, why))
        else:
            results.append((split, idx, label, x))
    return results, failures


def copy_selected(src_memmap, indices, out_path, tail):
    dst = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16,
                                    shape=(len(indices), *tail))
    bs = 512
    for s in range(0, len(indices), bs):
        dst[s:s + bs] = src_memmap[indices[s:s + bs]]
    dst.flush()
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='skeleton_data')
    ap.add_argument('--out_dir', default='data/ntu120_fall')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--chunksize', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='only first N files (smoke test)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = list_files(args.data_dir)
    if args.limit:
        files = files[:args.limit]
    n_files = len(files)
    print(f'Files found: {n_files}', flush=True)
    assert n_files > 0

    for f in files:
        f['fall'] = 1 if f['action'] == FALL_ACTION else 0
        f['split'] = 'train' if f['subject'] in TRAIN_SUBJECTS else 'test'
    n_train = sum(1 for f in files if f['split'] == 'train')
    n_test = n_files - n_train
    print(f'cross-subject split: train={n_train} '
          f'(falls={sum(f["fall"] for f in files if f["split"] == "train")}), '
          f'test={n_test} '
          f'(falls={sum(f["fall"] for f in files if f["split"] == "test")})', flush=True)

    tail = (N_FRAME_MAX, M, V, C)
    x_train = np.lib.format.open_memmap(os.path.join(args.out_dir, 'x_train.tmp.npy'),
                                        mode='w+', dtype=np.float16,
                                        shape=(n_train, *tail))
    x_test = np.lib.format.open_memmap(os.path.join(args.out_dir, 'x_test.tmp.npy'),
                                       mode='w+', dtype=np.float16,
                                       shape=(n_test, *tail))

    tasks = []
    it_tr = it_te = 0
    for f in files:
        idx = it_tr if f['split'] == 'train' else it_te
        tasks.append((idx, f['name'], f['fall'], f['split']))
        it_tr += (f['split'] == 'train')
        it_te += (f['split'] == 'test')

    chunks = [tasks[i:i + args.chunksize] for i in range(0, len(tasks), args.chunksize)]
    ok_tr, ok_te, failures = [], [], []
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for results, fails in pool.imap_unordered(
                process_chunk, ((c, args.data_dir) for c in chunks)):
            for (split, idx, label, x) in results:
                (x_train if split == 'train' else x_test)[idx] = x
                (ok_tr if split == 'train' else ok_te).append(idx)
            failures.extend(fails)
            done += len(results) + len(fails)
            rate = done / max(time.time() - t0, 1e-6)
            print(f'{done}/{n_files} parsed | fail={len(fails)} | {rate:.0f} files/s |'
                  f' ETA {(n_files - done) / max(rate, 1e-6) / 60:.1f} min', flush=True)

    with open(os.path.join(args.out_dir, 'manifest.csv'), 'w', newline='') as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(['filename', 'split', 'subject', 'camera', 'setup', 'replicate',
                     'action', 'fall'])
        for f in files:
            wr.writerow([f['name'], f['split'], f['subject'], f['cam'], f['setup'],
                         f['repl'], f['action'], f['fall']])
    with open(os.path.join(args.out_dir, 'failed_files.txt'), 'w') as ff:
        for name, why in failures:
            ff.write(f'{name}\t{why}\n')

    ok_tr = np.sort(np.array(ok_tr, dtype=int))
    ok_te = np.sort(np.array(ok_te, dtype=int))
    print('compacting memmaps ...', flush=True)
    copy_selected(x_train, ok_tr, os.path.join(args.out_dir, 'x_train.npy'), tail)
    copy_selected(x_test, ok_te, os.path.join(args.out_dir, 'x_test.npy'), tail)
    del x_train, x_test
    os.remove(os.path.join(args.out_dir, 'x_train.tmp.npy'))
    os.remove(os.path.join(args.out_dir, 'x_test.tmp.npy'))

    tr_map, te_map = set(ok_tr.tolist()), set(ok_te.tolist())
    y_train = np.array([t[2] for t in tasks if t[3] == 'train' and t[0] in tr_map],
                       dtype=np.uint8)
    y_test = np.array([t[2] for t in tasks if t[3] == 'test' and t[0] in te_map],
                      dtype=np.uint8)
    np.save(os.path.join(args.out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.out_dir, 'y_test.npy'), y_test)

    print(f'Done in {(time.time() - t0) / 60:.1f} min', flush=True)
    print(f'train kept {len(ok_tr)}/{n_train} | falls = {int(y_train.sum())}')
    print(f'test  kept {len(ok_te)}/{n_test} | falls = {int(y_test.sum())}')
    print(f'failed files: {len(failures)} -> failed_files.txt')


if __name__ == '__main__':
    main()
