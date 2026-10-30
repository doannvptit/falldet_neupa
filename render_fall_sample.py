"""Render a labelled NTU fall sample and check it with the finetuned model."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/falldet-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
import yaml

from data_gen.prepare_ntu120_fall_v2 import (
    build_sample, compact_and_fill, read_skeleton_sequence,
)
from feeders.feeder_fall import Feeder
from graph.ntu_rgb_d import joint_info
from utils import import_class, init_seed


def raw_tracks(frames, length):
    tracks = {}
    for t, frame in enumerate(frames):
        for body_id, joints in frame.items():
            if joints is not None:
                tracks.setdefault(body_id, np.zeros((len(frames), 25, 3), np.float32))[t] = joints
    ranked = []
    for track in tracks.values():
        valid = np.abs(track).sum(axis=(1, 2)) > 0
        energy = sum(track[valid, :, c].std() for c in range(3))
        ranked.append((energy, compact_and_fill(track)))
    ranked.sort(key=lambda item: -item[0])
    out = np.zeros((length, 2, 25, 3), np.float32)
    for m, (_, track) in enumerate(ranked[:2]):
        n = min(length, len(track))
        out[:n, m] = track[:n]
    return out


def setup_axis(ax, data, title):
    points = data[np.any(data != 0, axis=-1)]
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float((high - low).max()) * 0.58, 0.6)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius),
           xlabel='Horizontal', ylabel='Depth', zlabel='Vertical', title=title)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-65)
    ax.tick_params(labelsize=7)


def draw_pose(ax, pose, edges):
    artists = []
    for m, color in enumerate(('#1574c4', '#d57417')):
        points = pose[m]
        if not np.any(points):
            continue
        artists.append(ax.scatter(*points.T, color=color, s=14, depthshade=False))
        for a, b in edges:
            # The normalized spine joint is a valid joint exactly at the origin.
            artists.extend(ax.plot(*points[[a, b]].T, color=color, linewidth=2))
    return artists


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/ntu120_fall_v2')
    ap.add_argument('--raw', default='data/skeleton_extracted/nturgb+d_skeletons')
    ap.add_argument('--config', default='config/fall-detection/verify_2epoch.yaml')
    ap.add_argument('--weights', default='work_dir/fall/verify_3epoch/train/best/weights-2-17090.pt')
    ap.add_argument('--out', default='outputs/fall_sample')
    ap.add_argument('--index', type=int, help='Full test-array index; default: first labelled fall')
    ap.add_argument('--fps', type=int, default=15, help='Visualization playback rate, not source timestamp information')
    args = ap.parse_args()
    if args.fps <= 0:
        ap.error('--fps must be positive')
    init_seed(1024)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    x = np.load(data_dir/'x_test.npy', mmap_mode='r')
    labels = np.load(data_dir/'y_test.npy')
    index = int(np.flatnonzero(labels == 1)[0]) if args.index is None else args.index
    if not 0 <= index < len(labels) or labels[index] != 1:
        ap.error('Selected test index must have ground-truth fall label 1')
    failures = {line.split('\t')[0] for line in (data_dir/'failed_files.txt').read_text().splitlines()}
    with (data_dir/'manifest.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['split']=='test' and r['filename'] not in failures]
    assert len(rows) == len(labels)
    row = rows[index]
    assert int(row['fall']) == 1 and int(row['action']) == 43
    source = Path(args.raw)/row['filename']
    frames = read_skeleton_sequence(source)
    extracted = np.array(x[index], dtype=np.float32)
    rebuilt = build_sample(frames, extracted.shape[0])
    assert np.array_equal(rebuilt, extracted), 'Raw/array sample correspondence failed'
    valid = np.any(extracted != 0, axis=(1, 2, 3))
    length = int(np.flatnonzero(valid)[-1]+1)
    extracted = extracted[:length]
    raw = raw_tracks(frames, length)

    cfg = yaml.safe_load(Path(args.config).read_text())
    feeder_args = dict(cfg['test_feeder_args'], data_path=args.data, subsample_ratio=0, random_rot=False)
    # Parser.check_args copies these train modalities to the test feeder.
    # Mirror that behavior: Feeder's standalone multi_input default is True.
    for key in ('multi_input', 'bone', 'vel'):
        feeder_args[key] = cfg['train_feeder_args'].get(key, False)
    ds = Feeder(**feeder_args)
    tensor, label, _ = ds[index]
    model = import_class(cfg['model'])(**cfg['model_args'])
    model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).float())[-1]
        probability = float(torch.softmax(logits, dim=1)[0, 1])
    predicted = int(probability >= 0.5)
    print(f'{row["filename"]}: test index={index}, frames={length}, p(fall)={probability:.6f}', flush=True)

    # Camera XYZ has Y up. Keep the stored normalized axes unchanged on the right.
    raw_display = raw[..., [0, 2, 1]]
    display = extracted.copy()
    edges = joint_info([25, 10, 5])[1][0]
    fig = plt.figure(figsize=(12, 6.4), dpi=100)
    axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    setup_axis(axes[0], raw_display, 'Original 3D skeleton (camera coordinates)')
    setup_axis(axes[1], display, 'Extracted skeleton (normalized model data)')
    axes[1].set(xlabel='Normalized X', ylabel='Normalized Y', zlabel='Normalized Z')
    fig.suptitle(f'A043 FALL | TEST | {row["filename"]}', fontsize=12, fontweight='bold')
    fig.text(.5, .055, f'Ground truth: FALL (1) | Whole-clip prediction: {"FALL" if predicted else "NON-FALL"} | p(fall)={probability:.4f}', ha='center', fontsize=11)
    fig.text(.5, .022, 'Skeleton visualization, not RGB footage | Clip-level prediction, not a fall-onset timestamp', ha='center', fontsize=9, color='#555555')
    counter = fig.text(.5, .105, '', ha='center', fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, bottom=.13, top=.9, wspace=.05)
    artists = []
    writer = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=2200,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    video = out/'fall_skeleton.mp4'
    with writer.saving(fig, str(video), dpi=100):
        for t in range(length):
            for artist in artists:
                artist.remove()
            artists = draw_pose(axes[0], raw_display[t], edges) + draw_pose(axes[1], display[t], edges)
            counter.set_text(f'Frame {t+1}/{length} | Playback {args.fps} fps')
            writer.grab_frame()
            if t in (0, length//2, length-1):
                fig.savefig(out/f'frame_{t:03d}.png', dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4), dpi=120)
    for i, t in enumerate(np.linspace(0, length-1, 4, dtype=int)):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        setup_axis(ax, raw_display, f'Frame {t+1}/{length}')
        draw_pose(ax, raw_display[t], edges)
    fig.suptitle('A043 FALL — original 3D skeleton timeline', fontsize=13)
    fig.subplots_adjust(left=.02, right=.98, top=.86, bottom=.05, wspace=.15)
    fig.savefig(out/'fall_timeline.png')
    plt.close(fig)
    np.save(out/'sample_normalized.npy', extracted)
    np.save(out/'model_input.npy', tensor.unsqueeze(0).numpy())
    metadata = dict(source=str(source), prepared_data=str(data_dir/'x_test.npy'),
                    test_index=index, ground_truth=int(label), action='A043 falling',
                    raw_frames=len(frames), rendered_frames=length, playback_fps=args.fps,
                    playback_note='Chosen visualization rate; no source timestamps inferred',
                    model_input_shape=list(tensor.unsqueeze(0).shape), weights=args.weights,
                    predicted_label=predicted, p_fall=probability,
                    prediction_scope='Whole clip, using the same crop/resize as test feeder',
                    raw_matches_prepared_array=True, video=str(video))
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
