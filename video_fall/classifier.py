"""Use the same skeleton normalization/crop and fused binary head as training."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from data_gen.prepare_ntu120_fall_v2 import build_sample
from feeders.tools import valid_crop_resize
from utils import import_class
from .adapter import window_ends


def model_input(poses, p_interval=(.95,), window_size=64, coordinate_mode='xyz', projection=None):
    """Raw NTU25 (T,25,3); NaN rows mean an unobserved frame, not a normal pose."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (25, 3):
        raise ValueError('Expected (T,25,3) raw skeleton sequence')
    frames = []
    for p in poses:
        if not np.isfinite(p).all() or not np.any(p):
            frames.append({})
            continue
        # A hip-relative joint at (0,0,0) is valid, but the training converter
        # treats all-zero joints as missing. Shift the coordinate origin ONLY
        # in that case; spine centering cancels this constant immediately.
        # This numerical offset is not a camera-depth estimate.
        if np.any(np.all(p == 0, axis=-1)):
            p = p + np.array([0., 0., 4.], dtype=np.float32)
        frames.append({1: p})
    # Reuse the exact normalization and float16 round-trip used for NTU training.
    normalized = build_sample(frames, t_cap=150, coordinate_mode=coordinate_mode)
    if normalized is None:
        raise ValueError('Sequence contains no usable pose')
    if projection is not None:
        from pose_projection import project_pose
        normalized = project_pose(normalized, projection)
    data = normalized.astype(np.float32).transpose(3, 0, 2, 1)
    valid_frames = int(np.sum(np.abs(data).sum(axis=(0, 2, 3)) > 0)) if coordinate_mode == 'xy_zero_z' or projection is not None else int(np.sum(data.sum(0).sum(-1).sum(-1) != 0))
    return valid_crop_resize(data, valid_frames, p_interval, window_size)


def image_model_input(poses, p_interval=(.95,), window_size=64):
    """Pixels (T,25,2 or 3). Ignore any Z values, including NaN/inf depth."""
    from pose_2d import normalize_image_xy
    poses = np.asarray(poses, np.float32)
    if poses.ndim != 3 or poses.shape[1] != 25 or poses.shape[2] not in (2, 3):
        raise ValueError('Expected image XY sequence (T,25,2 or 3)')
    xyz = np.zeros((len(poses), 25, 3), np.float32)
    xyz[..., :2] = poses[..., :2]
    frames = [{1: p} if np.isfinite(p).all() and np.any(p) else {} for p in xyz]
    sample = build_sample(frames, t_cap=150, coordinate_mode='xy_zero_z')
    if sample is None:
        raise ValueError('No usable 2D image poses')
    data = sample.astype(np.float32).transpose(3, 0, 2, 1)
    valid = int(np.sum(np.abs(data).sum(axis=(0, 2, 3)) > 0))
    return normalize_image_xy(valid_crop_resize(data, valid, p_interval, window_size))


class BinaryClassifier:
    def __init__(self, config, weights, device='auto', threads=4):
        torch.set_num_threads(threads)
        cfg = yaml.safe_load(Path(config).read_text())
        if cfg['model_args']['num_class'] != 2:
            raise ValueError('Expected a binary model configuration')
        if any(cfg['train_feeder_args'].get(k, False) for k in ('multi_input', 'bone', 'vel')):
            raise ValueError('This adapter requires joint input with internal motion')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = import_class(cfg['model'])(**cfg['model_args']).to(self.device)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.eval()
        self.p_interval = cfg['test_feeder_args'].get('p_interval', [.95])
        self.window_size = cfg['test_feeder_args']['window_size']
        self.coordinate_mode = 'xy_zero_z' if cfg['test_feeder_args'].get('zero_z', False) else 'xyz'
        self.projection = None
        projection_path = cfg['test_feeder_args'].get('projection_path')
        if projection_path:
            from pose_projection import load_projection
            self.projection = load_projection(projection_path)
            self.coordinate_mode = 'xyz'
        if cfg['test_feeder_args'].get('image2d_multiview', False):
            if self.projection is not None:
                raise ValueError('Image 2D inference must not use PCA')
            self.coordinate_mode = 'image2d_multiview'

    @torch.inference_mode()
    def score(self, poses):
        if self.coordinate_mode == 'image2d_multiview':
            x = image_model_input(poses, self.p_interval, self.window_size)
        else:
            x = model_input(poses, self.p_interval, self.window_size, self.coordinate_mode, self.projection)
        output = self.model(torch.from_numpy(x[None]).to(self.device))
        logits = output[-1] if isinstance(output, tuple) else output
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())


def classify_records(records, classifier, fps, window=64, stride=8,
                     min_frames=16, max_gap=8, min_coverage=.6, threshold=.5):
    tracks = defaultdict(list)
    for record in records:
        tracks[record['track_id']].append(record)
    results = []
    for tid, track in sorted(tracks.items()):
        track.sort(key=lambda r: r['frame'])
        segments = [[]]
        for record in track:
            if segments[-1] and record['frame'] - segments[-1][-1]['frame'] > max_gap + 1:
                segments.append([])
            segments[-1].append(record)
        for segment in segments:
            first, last = segment[0]['frame'], segment[-1]['frame']
            by_frame = {r['frame']: r['ntu25'] for r in segment}
            for end in window_ends(first, last, min_frames, stride):
                start = max(first, end - window + 1)
                observed = sum(f in by_frame for f in range(start, end + 1))
                coverage = observed / (end - start + 1)
                if observed < min_frames or coverage < min_coverage:
                    continue
                poses = np.stack([by_frame.get(f, np.full((25, 3), np.nan, np.float32))
                                  for f in range(start, end + 1)])
                probability = classifier.score(poses)
                results.append(dict(track_id=tid, start_frame=start, end_frame=end,
                                    start_sec=start/fps, end_sec=end/fps,
                                    observed_frames=observed, coverage=coverage,
                                    p_fall=probability, predicted_label=int(probability >= threshold)))
    return sorted(results, key=lambda r: (r['end_frame'], r['track_id']))
