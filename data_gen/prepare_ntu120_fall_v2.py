#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare NTU RGB+D 120 skeletons for binary FALL detection finetune (v2, MS-G3D faithful).

Differences from prepare_ntu120_fall.py (v1):
  1. Official NTU RGB+D 120 CROSS-SUBJECT split (53 training subjects, as defined
     in the NTU RGB+D 120 paper arXiv:1905.04757 and implemented in MS-G3D).
  2. Files listed in NTU_RGBD120_samples_with_missing_skeletons.txt are excluded.
  3. Each body track is shifted to the front (head-shift, MS-G3D style) and
     internal missing frames are hold-filled, so that the feeder's
     `valid_frame_num` counting is always correct (no all-zero crops).
  4. MS-G3D `pre_normalization` is applied: subtract the per-frame spine joint
     (#1) of person-0, rotate hip->spine bone of person-0's first frame to the z
     axis, then the right/left-shoulder bone (joints 8/4) to the x axis. This
     keeps the input distribution as close as possible to the data the
     pretrained NeuroPath (EGGCN) checkpoint was trained on.
  5. Sequences are trimmed to T_CAP=150 frames (99.9% of falling samples used <=
     139 frames), float16 storage.

Output (data/ntu120_fall_v2/):
  x_train.npy (N, T_CAP, 2, 25, 3) float16   [ T, M(body), V(joint), C(xyz) ]
  x_test.npy  (N, T_CAP, 2, 25, 3) float16
  y_train.npy / y_test.npy (N,) uint8 -> 1 = falling (A043), 0 = not falling
  manifest.csv / failed_files.txt / stats.json
"""
import argparse
import csv
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

FALL_ACTION = 43                       # NTU RGB+D 120 action id for "falling"
NUM_JOINT = V = 25
NUM_CH = C = 3
MAX_BODY_READ = 4
MAX_BODY_KEEP = M = 2
DEFAULT_T_CAP = 150

FILE_RE = re.compile(r'^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\.skeleton$')

# Official NTU RGB+D 120 cross-subject training subjects (MS-G3D / NTU120 paper)
NTU120_TRAINING_SUBJECTS = frozenset([
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
])


# --------------------------------------------------------------------------- #
# skeleton parsing
# --------------------------------------------------------------------------- #
def read_skeleton_sequence(path):
    """Parse a .skeleton file.

    Returns list over frames; each frame is dict bodyID -> (V, C) float32
    (or None when that body-in-frame is invalid)."""
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
            body_id = lines[p].split()[0]
            p += 1
            if p >= total:
                break
            try:
                joint_count = int(lines[p].split()[0])
            except (ValueError, IndexError):
                break
            p += 1
            joints = np.zeros((V, C), dtype=np.float32)
            ok = True
            for v in range(min(joint_count, V)):
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
            if not ok or np.isnan(joints).any() \
                    or (joints == 0).all() or (np.abs(joints) > 1e6).any():
                bodies[body_id] = None        # invalid body in frame
            else:
                bodies[body_id] = joints
        frames.append(bodies)
    return frames


# --------------------------------------------------------------------------- #
# MS-G3D pre_normalization equivalents
# --------------------------------------------------------------------------- #
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, ab, ac, ad = a * a, a * b, a * c, a * d
    bb, bc, bd = b * b, b * c, b * d
    cc, cd = c * c, c * d
    dd = d * d
    return np.array([[aa + bb - cc - dd, 2 * (ab + cd), 2 * (ac - bd)],
                     [2 * (ab - cd), aa + cc - bb - dd, 2 * (bc + ad)],
                     [2 * (ac + bd), 2 * (bc - ad), aa + dd - bb - cc]],
                    dtype=np.float32)


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))


def body_energy(s):
    """MS-G3D style body energy: std of each xyz channel over valid frames."""
    valid = np.abs(s).sum(-1).sum(-1) != 0
    s = s[valid]
    if len(s) == 0:
        return 0.0
    return float(s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std())


def compact_and_fill(arr):
    """Shift a body track to the front (drop the invalid head), hold-fill
    internal invalid frames with the previous valid pose. The track ends right
    after the last valid frame (anything behind stays zero when copied out).
    Returns the contiguous float32 array or None."""
    s = np.array(arr, dtype=np.float32)                     # (T, V, C)
    valid = np.abs(s).sum(-1).sum(-1) > 0                   # (T,)
    if not valid.any():
        return None
    first = int(valid.argmax())
    last = int(np.nonzero(valid)[0][-1])
    s = s[first:last + 1].copy()                            # real span
    cur = s[0].copy()
    for t in range(len(s)):
        if valid[first + t]:
            cur = s[t]
        else:                                               # internal gap -> hold
            s[t] = cur
    return s


def pre_normalization_inplace(data, T):
    """data: (M, T, V, C). MS-G3D normalization on the (already buena) tracks:
    per-frame spine-center subtraction + two rotations fixed by person-0/f0."""
    # ---- subtract person-0's per-frame center joint #1 (spine mid) ----
    main_center = np.array(data[0, :, 1:2, :])              # (T, 1, C)
    for m in range(M):
        person = np.array(data[m])
        mask = (np.abs(person).sum(-1) > 0).astype(np.float32).reshape(T, V, 1)
        data[m] = (person - main_center) * mask
    # ---- rotate hip(jpt 0) -> spine(jpt 1) bone of person 0 frame 0 to z axis
    v = data[0, 0, 1] - data[0, 0, 0]
    if np.linalg.norm(v) > 1e-6:
        axis = np.cross(v, [0, 0, 1])
        if np.linalg.norm(axis) > 1e-6:
            matrix_z = rotation_matrix(axis, angle_between(v, [0, 0, 1]))
            for m in range(M):
                data[m] = data[m] @ matrix_z.T
    # ---- rotate right(8)/left(4) shoulder bone of person 0 frame 0 to x axis
    v = data[0, 0, 8] - data[0, 0, 4]
    if np.linalg.norm(v) > 1e-6:
        axis = np.cross(v, [1, 0, 0])
        if np.linalg.norm(axis) > 1e-6:
            matrix_x = rotation_matrix(axis, angle_between(v, [1, 0, 0]))
            for m in range(M):
                data[m] = data[m] @ matrix_x.T
    # rotations are linear maps: zero frames map to zero frames (no mask needed)
    return data


def build_sample(frames, t_cap, coordinate_mode='xyz'):
    """frames -> (t_cap, M, V, C) float16, or None if unusable."""
    if coordinate_mode not in ('xyz', 'xy_zero_z', 'camera_xyz'):
        raise ValueError(f'Unknown coordinate mode: {coordinate_mode}')
    if not frames:
        return None

    # per-body (25,3) tracks over the whole timeline; invalid -> None rows
    body_tracks = {}
    for t, fr in enumerate(frames):
        for b, j in fr.items():
            if j is None:
                continue
            body_tracks.setdefault(b, [None] * len(frames))[t] = j
    if not body_tracks:
        return None

    # choose top-2 bodies by MS-G3D energy, ordered by energy desc
    blobs = []
    for b, track in body_tracks.items():
        arr = np.zeros((len(track), V, C), dtype=np.float32)
        for t, j in enumerate(track):
            if j is not None:
                arr[t] = j
        valid = np.abs(arr).sum(-1).sum(-1) != 0
        if not valid.any():
            continue
        blobs.append((float(arr[valid][:, :, 0].std()
                            + arr[valid][:, :, 1].std()
                            + arr[valid][:, :, 2].std()),
                      (arr, valid)))
    if not blobs:
        return None
    blobs.sort(key=lambda x: -x[0])

    T_raw = len(frames)
    data = np.zeros((M, T_raw, V, C), dtype=np.float32)
    lengths = []
    for m, (energy, (arr, valid)) in enumerate(blobs[:MAX_BODY_KEEP]):
        comp = compact_and_fill(arr)
        if comp is None:
            continue
        data[m, :len(comp)] = comp
        lengths.append(len(comp))
    if not lengths:
        return None

    T = min(max(lengths), t_cap)                # usable trimmed length
    data = data[:, :T]

    # per-body zero spans beyond their own (compacted) length stay zero
    for m, (energy, (arr, valid)) in enumerate(blobs[:MAX_BODY_KEEP]):
        comp = compact_and_fill(arr)
        if comp is None or len(comp) >= T:
            continue
        data[m, len(comp):] = 0.0

    if coordinate_mode in ('xy_zero_z', 'camera_xyz'):
        # Preserve the camera axes; camera_xyz keeps depth only to synthesize
        # camera views at train time. Do not apply body-dependent rotations.
        mask = np.abs(data).sum(-1, keepdims=True) > 0
        if coordinate_mode == 'xy_zero_z':
            data[..., 2] = 0
        center = data[0, :, 1:2, :].copy()
        data = (data - center[None]) * mask
        if coordinate_mode == 'xy_zero_z':
            data[..., 2] = 0
    else:
        pre_normalization_inplace(data, T)

    out = np.zeros((t_cap, M, V, C), dtype=np.float16)
    out[:T] = data.transpose(1, 0, 2, 3).astype(np.float16)
    if not np.isfinite(out.astype(np.float32)).all():
        return None
    return out


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
DATA_DIR = None
T_CAP_G = DEFAULT_T_CAP
COORDINATE_MODE_G = 'xyz'


def process_chunk(chunk):
    results, failures = [], []
    if isinstance(chunk, tuple) and len(chunk) == 2:
        chunk, data_dir = chunk
    for (idx, fname, label, split) in chunk:
        x = None
        try:
            frames = read_skeleton_sequence(os.path.join(DATA_DIR, fname))
            x = build_sample(frames, T_CAP_G, COORDINATE_MODE_G)
            why = 'empty / invalid skeletons'
        except Exception as exc:
            why = repr(exc)[:160]
        if x is None:
            failures.append((fname, why))
        else:
            results.append((split, idx, label, x))
    return results, failures


# --------------------------------------------------------------------------- #
def main():
    global DATA_DIR, T_CAP_G, COORDINATE_MODE_G
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='skeleton_data')
    ap.add_argument('--out_dir', default='data/ntu120_fall_v2')
    ap.add_argument('--missing_list', default='data_gen/NTU_RGBD120_samples_with_missing_skeletons.txt')
    ap.add_argument('--t_cap', type=int, default=DEFAULT_T_CAP)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunksize', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0, help='only first N files (raw prefix)')
    ap.add_argument('--smoke', action='store_true',
                    help='stratified mini subset: fall_sample falling + neg_sample negatives (both classes in both splits)')
    ap.add_argument('--fall_sample', type=int, default=200)
    ap.add_argument('--neg_sample', type=int, default=600)
    ap.add_argument('--sample_seed', type=int, default=1024)
    ap.add_argument('--coordinate-mode', choices=('xyz', 'xy_zero_z', 'camera_xyz'), default='xyz')
    args = ap.parse_args()

    guard = os.path.join(args.out_dir, '.protected')
    if os.path.exists(guard):
        print(f'[guard] {args.out_dir} has .protected marker -> dataset is '
              'finalised; refusing to rebuild (delete the marker explicitly '
              'if you want to regenerate).', flush=True)
        return

    DATA_DIR, T_CAP_G = args.data_dir, args.t_cap
    COORDINATE_MODE_G = args.coordinate_mode
    os.makedirs(args.out_dir, exist_ok=True)

    files = []
    for name in sorted(os.listdir(args.data_dir)):
        mm = FILE_RE.match(name)
        if mm is None:
            continue
        setup, cam, subject, repl, action = map(int, mm.groups())
        files.append(dict(name=name, subject=subject, cam=cam, setup=setup,
                          repl=repl, action=action))
    n_all = len(files)

    ignored = set()
    if args.missing_list and os.path.isfile(args.missing_list):
        with open(args.missing_list) as f:
            names = {line.strip() for line in f if line.strip()}
        # entries may come with or without the .skeleton extension
        ignored = set(names) | {n + '.skeleton' for n in names}
    files = [f for f in files if f['name'] not in ignored]
    n_ignored = n_all - len(files)
    print(f'files found {n_all}, ignored (missing-skeletons) {n_ignored}, '
          f'using {len(files)}', flush=True)

    if args.limit:
        files = files[:args.limit]
    elif args.smoke:
        import random
        rng = random.Random(args.sample_seed)
        falls = [f for f in files if f['action'] == FALL_ACTION]
        negs = [f for f in files if f['action'] != FALL_ACTION]
        rng.shuffle(falls)
        rng.shuffle(negs)
        files = sorted(falls[:args.fall_sample] + negs[:args.neg_sample],
                       key=lambda f: f['name'])

    for f in files:
        f['fall'] = 1 if f['action'] == FALL_ACTION else 0
        f['split'] = 'train' if f['subject'] in NTU120_TRAINING_SUBJECTS else 'test'
    n_tr = sum(1 for f in files if f['split'] == 'train')
    fall_tr = sum(f['fall'] for f in files if f['split'] == 'train')
    fall_te = sum(f['fall'] for f in files if f['split'] == 'test')
    print(f'official NTU120 cross-subject split: train={n_tr} (falls={fall_tr}), '
          f'test={len(files) - n_tr} (falls={fall_te})', flush=True)

    tasks = []
    it_tr = it_te = 0
    for f in files:
        idx = it_tr if f['split'] == 'train' else it_te
        tasks.append((idx, f['name'], f['fall'], f['split']))
        it_tr += (f['split'] == 'train')
        it_te += (f['split'] == 'test')

    tail = (args.t_cap, M, V, C)
    x_tr = np.lib.format.open_memmap(os.path.join(args.out_dir, 'x_train.tmp.npy'),
                                     mode='w+', dtype=np.float16, shape=(n_tr, *tail))
    x_te = np.lib.format.open_memmap(os.path.join(args.out_dir, 'x_test.tmp.npy'),
                                     mode='w+', dtype=np.float16,
                                     shape=(len(tasks) - n_tr, *tail))

    chunks = [tasks[i:i + args.chunksize] for i in range(0, len(tasks), args.chunksize)]
    keep_tr, keep_te, failures = [], [], []
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for results, fails in pool.imap_unordered(process_chunk, chunks):
            for (split, idx, label, x) in results:
                (x_tr if split == 'train' else x_te)[idx] = x
                (keep_tr if split == 'train' else keep_te).append((idx, f'{{}}{split}', label))
            failures.extend(fails)
            done += len(results) + len(fails)
            rate = done / max(time.time() - t0, 1e-6)
            print(f'{done}/{len(tasks)} parsed | fail={len(fails)} | {rate:.0f} files/s'
                  f' | ETA {(len(tasks) - done) / max(rate, 1e-6) / 60:.1f} min', flush=True)

    # compact the memmaps into the final arrays
    keep_tr_idx = np.sort(np.array([k[0] for k in keep_tr], dtype=int))
    keep_te_idx = np.sort(np.array([k[0] for k in keep_te], dtype=int))
    print('compacting memmaps ...', flush=True)

    def compact(src, sel, dst):
        dst = np.lib.format.open_memmap(dst, mode='w+', dtype=np.float16,
                                        shape=(len(sel), *tail))
        bs = 128          # ~11 MB per chunk; flushed after each so the kernel
        for s in range(0, len(sel), bs):      # never accumulates endless dirty pages
            dst[s:s + bs] = src[sel[s:s + bs]]
            dst.flush()
        dst.flush()
        return dst

    compact(x_tr, keep_tr_idx, os.path.join(args.out_dir, 'x_train.npy'))
    compact(x_te, keep_te_idx, os.path.join(args.out_dir, 'x_test.npy'))
    del x_tr, x_te
    for tmp_name in ('x_train.tmp.npy', 'x_test.tmp.npy'):
        tmp_path = os.path.join(args.out_dir, tmp_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    keep_tr_set, keep_te_set = set(keep_tr_idx.tolist()), set(keep_te_idx.tolist())
    y_tr = np.array([t[2] for t in tasks
                     if t[3] == 'train' and t[0] in keep_tr_set],
                    dtype=np.uint8)
    y_te = np.array([t[2] for t in tasks
                     if t[3] == 'test' and t[0] in keep_te_set],
                    dtype=np.uint8)
    np.save(os.path.join(args.out_dir, 'y_train.npy'), y_tr)
    np.save(os.path.join(args.out_dir, 'y_test.npy'), y_te)

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

    stats = dict(n_all=n_all, ignored=n_ignored, used=len(files),
                 train=len(y_tr), test=len(y_te),
                 falls_train=int(y_tr.sum()), falls_test=int(y_te.sum()),
                 failures=len(failures), t_cap=args.t_cap,
                 coordinate_mode=args.coordinate_mode)
    with open(os.path.join(args.out_dir, 'stats.json'), 'w') as fj:
        json.dump(stats, fj, indent=2)
    print('Done in %.1f min' % ((time.time() - t0) / 60), flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
