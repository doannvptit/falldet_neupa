"""NumPy-only equivalent of the image-XY training/inference preprocessing."""
import numpy as np


def image_model_input(poses):
    """Pixel XY (T,25,2/3), one tracked person -> float32 (3,64,25,2).

    Z is ignored. A missing frame is all-zero or contains nonfinite XY.
    Match training: compact, hold-fill, spine center, float16 roundtrip,
    95% center crop, align_corners=False resize, median RMS normalization.
    """
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1] != 25 or poses.shape[2] not in (2, 3):
        raise ValueError('Expected (T,25,2 or 3) in pixels, NTU25 joint order')
    xyz = np.zeros((len(poses), 25, 3), np.float32)
    xyz[..., :2] = poses[..., :2]
    valid = np.isfinite(xyz).all(axis=(1, 2)) & np.any(xyz != 0, axis=(1, 2))
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError('No usable image XY pose')
    first, last = int(indices[0]), int(indices[-1])
    track = xyz[first:last+1].copy()
    for t in range(1, len(track)):
        if not valid[first+t]:
            track[t] = track[t-1]
    track = track[:150]
    mask = np.any(track != 0, axis=-1, keepdims=True)
    track = (track - track[:, 1:2]) * mask
    with np.errstate(over='ignore'):
        track = track.astype(np.float16).astype(np.float32)
    if not np.isfinite(track).all():
        raise ValueError('XY coordinates overflow training float16 representation')
    data = np.zeros((3, 150, 25, 2), np.float32)
    data[:, :len(track), :, 0] = track.transpose(2, 0, 1)
    count = int(np.sum(np.abs(data).sum(axis=(0, 2, 3)) > 0))
    bias = int((1-.95)*count/2)
    cropped = data[:, bias:count-bias]
    length = cropped.shape[1]
    if not length:
        raise ValueError('No non-degenerate image XY pose')
    # PyTorch bilinear interpolation leaves the C*V*M dimension unchanged.
    # Only its temporal axis needs interpolation, with half-pixel coordinates.
    pos = (np.arange(64, dtype=np.float32)+np.float32(.5))*np.float32(length/64)-np.float32(.5)
    pos = np.maximum(pos, 0)
    lo = pos.astype(np.int64)
    hi = np.minimum(lo+1, length-1)
    weight = (pos-lo).astype(np.float32)[None, :, None, None]
    resized = cropped[:, lo]*(1-weight) + cropped[:, hi]*weight
    return normalize_image_xy(resized)


def normalize_image_xy(data):
    data = np.asarray(data, dtype=np.float32)
    out = np.zeros_like(data)
    valid = np.any(data[:2] != 0, axis=0)
    center = data[:2, :, 1:2, 0:1].copy()
    for t in np.flatnonzero(~valid[:, :, 0].any(axis=1)):
        people = np.flatnonzero(valid[t].any(axis=0))
        if len(people):
            center[:, t, 0, 0] = data[:2, t, 1, people[0]]
    out[:2] = (data[:2]-center)*valid[None]
    squared = (out[:2]**2).sum(0)
    counts = valid.sum(axis=(1, 2))
    radii = np.sqrt(squared.sum(axis=(1, 2))/np.maximum(counts, 1))
    usable = radii[radii > 1e-6]
    if not len(usable):
        raise ValueError('No non-degenerate image XY pose')
    out[:2] /= max(float(np.median(usable)), 1e-6)
    return out
