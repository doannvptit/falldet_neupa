"""Approximate Halpe26 pixel XY -> Kinect/NTU25; no depth estimation.

Source layout: MMPose configs/_base_/datasets/halpe26.py (zero-based).
Hands/thumbs are geometric proxies, not observations. Confidence propagates
from every required source joint; missing source joints produce zero output.
"""
import numpy as np

ADAPTER_VERSION = 'halpe26_to_ntu25_v1'
DIRECT = {4:5, 5:7, 6:9, 8:6, 9:8, 10:10,
          12:11, 13:13, 14:15, 16:12, 17:14, 18:16}
SYNTHETIC_JOINTS = [1, 7, 11, 20, 21, 22, 23, 24]


def map_halpe26(points, scores, min_confidence=.3):
    """Return (25,3) XYZ in pixels, Z=0, and (25,) confidence.

    Threshold applies to source scores, before reducing proxy confidence.
    Head/neck/central hip use Halpe predictions with torso/ear fallbacks.
    Foot uses the mean of visible big/small toes. Missing joints stay zero.
    """
    points = np.asarray(points, np.float32)
    scores = np.asarray(scores, np.float32)
    if points.shape != (26,2) or scores.shape != (26,):
        raise ValueError('Expected Halpe26 XY (26,2) and scores (26,)')
    if not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence must be in [0,1]')
    good = np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= min_confidence)
    xy = np.zeros((25,2), np.float32)
    confidence = np.zeros(25, np.float32)

    def combine(indices, weights=None):
        if not all(good[i] for i in indices):
            return np.zeros(2,np.float32), 0.
        return (np.mean(points[indices],axis=0) if weights is None else
                np.asarray(weights,np.float32) @ points[indices]), float(np.min(scores[indices]))

    for target, source in DIRECT.items():
        if good[source]:
            xy[target], confidence[target] = points[source], scores[source]
    xy[0], confidence[0] = combine([19]) if good[19] else combine([11,12])
    xy[3], confidence[3] = combine([17]) if good[17] else combine([3,4])
    xy[20], confidence[20] = combine([5,6])
    if confidence[0] and confidence[20]:
        xy[1] = (xy[0]+xy[20])/2
        confidence[1] = min(confidence[0],confidence[20])
    if good[18]:
        xy[2], confidence[2] = points[18], scores[18]
    elif confidence[20] and confidence[3]:
        xy[2] = .75*xy[20]+.25*xy[3]
        confidence[2] = min(confidence[20],confidence[3])
    for target, sources in [(15,[20,22]),(19,[21,23])]:
        visible = [i for i in sources if good[i]]
        if visible:
            xy[target], confidence[target] = combine(visible)
    for wrist, elbow, hand, tip, thumb in [(9,7,7,21,22),(10,8,11,23,24)]:
        if good[wrist] and good[elbow]:
            direction = points[wrist]-points[elbow]
            for target, factor in [(hand,.15),(tip,.30),(thumb,.10)]:
                xy[target] = points[wrist]+factor*direction
                confidence[target] = .5*min(scores[wrist],scores[elbow])
    output = np.zeros((25,3), np.float32)
    output[:,:2] = xy
    return output, confidence
