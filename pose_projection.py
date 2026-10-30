"""Fixed train-only PCA projection of normalized NTU XYZ into XY with Z=0."""
import numpy as np


def load_projection(path):
    with np.load(path, allow_pickle=False) as p:
        mean = np.asarray(p['mean'], np.float32)
        components = np.asarray(p['components'], np.float32)
    if mean.shape != (3,) or components.shape != (2, 3):
        raise ValueError('Projection must contain mean (3,) and components (2,3)')
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError('Projection contains non-finite values')
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-5)
    return mean, components


def project_pose(data, projection):
    """Last axis XYZ. Keep all-zero padding/joints zero; never mutate source."""
    mean, components = projection
    data = np.asarray(data, np.float32)
    valid = np.any(data != 0, axis=-1)
    out = np.zeros_like(data)
    out[..., :2] = ((data - mean) @ components.T) * valid[..., None]
    return out
