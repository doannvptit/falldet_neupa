"""Causal tracked-person windows; hold-fill only within a short-gap segment."""
from collections import defaultdict
import numpy as np
from .adapter import window_ends


def classify_records(records, classifier, fps, window=64, stride=8,
                     min_frames=16, max_gap=8, min_coverage=.6, threshold=.5,
                     require_full_window=False):
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
                if require_full_window and end-start+1<window:
                    continue
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
