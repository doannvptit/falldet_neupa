"""Virtual cameras for training; image XY normalization for depth-free inference."""
import numpy as np

EVAL_CAMERAS = {'overhead': (45., 80.), 'side_left': (0., 15.), 'side_right': (180., 15.)}


def virtual_camera(data, view=None, rng=None):
    """Project centered camera-space NTU (T,M,V,3), camera Y points up.

    One camera is held fixed for the entire sequence. Output X right, Y down.
    Train mixture: 50% overhead, 20% left, 20% right, 10% other oblique views.
    """
    rng = np.random if rng is None else rng
    if view is None:
        group = rng.choice(['overhead', 'side_left', 'side_right', 'oblique'], p=[.5, .2, .2, .1])
        if group == 'overhead':
            azimuth, elevation = rng.uniform(-180, 180), rng.uniform(60, 88)
        elif group == 'oblique':
            azimuth, elevation = rng.uniform(-180, 180), rng.uniform(0, 60)
        else:
            azimuth = (0 if group == 'side_left' else 180) + rng.uniform(-35, 35)
            elevation = rng.uniform(0, 35)
        distance = rng.uniform(3, 7) if rng.random() < .7 else None
    else:
        azimuth, elevation = EVAL_CAMERAS[view]
        distance = 5.
    az, el = np.deg2rad([azimuth, elevation])
    right = np.array([-np.sin(az), 0., np.cos(az)], np.float32)
    up = np.array([-np.sin(el)*np.cos(az), np.cos(el), -np.sin(el)*np.sin(az)], np.float32)
    toward_camera = np.array([np.cos(el)*np.cos(az), np.sin(el), np.cos(el)*np.sin(az)], np.float32)
    valid = np.any(data != 0, axis=-1)
    out = np.zeros_like(data, dtype=np.float32)
    out[..., 0] = data @ right
    out[..., 1] = -(data @ up)
    if distance is not None:
        depth = np.clip(distance - data @ toward_camera, distance*.35, distance*2)
        out[..., :2] *= (distance/depth)[..., None]
    out *= valid[..., None]
    return out


def normalize_image_xy(data):
    """(C,T,V,M) -> centered, scale invariant image XY with exact zero Z.

    Center on person-0 spine-mid (or the remaining person) every frame and
    use one median RMS body radius for the causal window. No depth is used.
    """
    data = np.asarray(data, dtype=np.float32)
    out = np.zeros_like(data)
    valid = np.any(data[:2] != 0, axis=0)  # T,V,M; ignore depth entirely
    center = data[:2, :, 1:2, 0:1].copy()
    # A crop can contain only person-1 after person-0's track has ended.
    for t in np.flatnonzero(~valid[:, :, 0].any(axis=1)):
        people = np.flatnonzero(valid[t].any(axis=0))
        if len(people):
            center[:, t, 0, 0] = data[:2, t, 1, people[0]]
    out[:2] = (data[:2] - center) * valid[None]
    squared = (out[:2]**2).sum(0)
    counts = valid.sum(axis=(1, 2))
    radii = np.sqrt(squared.sum(axis=(1, 2))/np.maximum(counts, 1))
    usable = radii[radii > 1e-6]
    if not len(usable):
        raise ValueError('No non-degenerate image XY pose')
    out[:2] /= max(float(np.median(usable)), 1e-6)
    return out


def augment_image_xy(data, rng=None):
    rng = np.random if rng is None else rng
    out = data.copy()
    valid = np.any(out[:2] != 0, axis=0)
    # Simulate mirrored cameras, pixel jitter, and short joint occlusions.
    if rng.random() < .5:
        out[0] *= -1
    out[:2] += rng.normal(0, .015, out[:2].shape).astype(np.float32) * valid[None]
    if rng.random() < .5:
        for _ in range(int(rng.randint(1, 4))):
            joint = int(rng.randint(2, 25))
            begin = int(rng.randint(0, out.shape[1]))
            length = int(rng.randint(2, 9))
            out[:, begin:begin+length, joint] = 0
    out[2] = 0
    return out
