"""Explicit approximate MediaPipe-33 -> Kinect/NTU-25 joint mapping."""
import numpy as np

# Rows are NTU's zero-based joint order. Synthetic joints are weighted averages.
JOINT_SOURCES = (
    ((23, .5), (24, .5)),                         # SpineBase
    ((23, .25), (24, .25), (11, .25), (12, .25)), # SpineMid
    ((11, .375), (12, .375), (7, .125), (8, .125)), # Neck
    ((7, .5), (8, .5)),                          # Head: ear midpoint
    ((11, 1.),), ((13, 1.),), ((15, 1.),),
    ((15, .5), (19, .5)),                        # Left hand center
    ((12, 1.),), ((14, 1.),), ((16, 1.),),
    ((16, .5), (20, .5)),                        # Right hand center
    ((23, 1.),), ((25, 1.),), ((27, 1.),), ((31, 1.),),
    ((24, 1.),), ((26, 1.),), ((28, 1.),), ((32, 1.),),
    ((11, .5), (12, .5)),                        # SpineShoulder
    ((19, 1.),), ((21, 1.),), ((20, 1.),), ((22, 1.),),
)
MAPPING = np.zeros((25, 33), dtype=np.float32)
for joint, sources in enumerate(JOINT_SOURCES):
    for source, weight in sources:
        MAPPING[joint, source] = weight


def map_landmarks(points):
    """(...,33,3) image landmarks -> approximate NTU25. Z is always ignored."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-2:] != (33, 3) or not np.isfinite(points[..., :2]).all():
        raise ValueError('Expected finite image XY (...,33,3) landmarks')
    image = np.zeros_like(points)
    image[..., :2] = points[..., :2]
    return np.einsum('jk,...kc->...jc', MAPPING, image)


def map_visibility(visibility):
    visibility = np.asarray(visibility, dtype=np.float32)
    if visibility.shape != (33,):
        raise ValueError('Expected 33 visibility values')
    return np.array([min(visibility[i] for i, _ in sources)
                     for sources in JOINT_SOURCES], dtype=np.float32)


def window_ends(start, end, min_frames=16, stride=8):
    """Inclusive causal endpoints, with an explicit EOF/tail endpoint."""
    if min_frames < 1 or stride < 1:
        raise ValueError('min_frames and stride must be positive')
    if end - start + 1 < min_frames:
        return []
    ends = list(range(start + min_frames - 1, end + 1, stride))
    if ends[-1] != end:
        ends.append(end)
    return ends


class PoseTracker:
    """Greedy image-center association with a distance gate and short expiry.

    IDs are independent of MediaPipe result ordering. This lightweight tracker
    can still swap identities during overlap/occlusion; it is not a ReID model.
    """
    def __init__(self, max_gap=8, max_distance=.2):
        self.max_gap = max_gap
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 1

    def update(self, frame, poses):
        return self.update_centers(frame,[p[[23,24],:2].mean(axis=0) for p in poses])

    def update_centers(self, frame, centers):
        """Associate normalized image XY centers independent of joint layout."""
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if frame - t[0] <= self.max_gap + 1}
        centers = [np.asarray(p,dtype=np.float32) for p in centers]
        candidates = sorted((float(np.linalg.norm(center - old_center)), i, tid)
                            for i, center in enumerate(centers)
                            for tid, (_, old_center) in self.tracks.items())
        used, matched, ids = set(), set(), [None] * len(centers)
        for distance, i, tid in candidates:
            if distance <= self.max_distance and i not in used and tid not in matched:
                ids[i] = tid
                used.add(i)
                matched.add(tid)
        for i, center in enumerate(centers):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = (frame, center)
        return ids
