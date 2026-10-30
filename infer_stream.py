"""Live RTSP stream -> YOLOX-s -> RTMPose-m Halpe26 -> NTU25 -> fall ONNX.

Single-pass live pipeline: decode -> detect -> pose -> track -> causal window
-> overlay. infer_video.py cannot take a live source because it re-opens the
input for a second render pass, which a live stream cannot replay. Here the
annotation is produced while the frame is decoded, so an RTSP URL works
directly.

Window scoring mirrors fall_onnx.windows.classify_records: same window, stride,
min_frames, max_gap and coverage rules, and the same explicit endpoint at the
end of a segment. The only difference is causal timing - a window ending at
frame f is scored as soon as frame f arrives, not after the whole track ends.
.venv/bin/python infer_stream.py --stream 'rtsp://admin:LumiVn@2021@10.10.30.86:554/ch1/0' --name probe2 --device cuda --duration 180 --preview --preview-scale 0.5 --no-render --output-dir outputs/stream
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# The OpenCV FFMPEG backend defaults to UDP for RTSP, which drops frames on a
# busy link; TCP keeps the stream ordered. timeout bounds the TCP connect.
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp|timeout;5000000')

import cv2
from fall_onnx.adapter import PoseTracker
from fall_onnx.halpe26 import map_halpe26, ADAPTER_VERSION
from fall_onnx.pose import PoseEstimator, DEFAULT_DETECTOR, DEFAULT_POSE
from fall_onnx.classifier import BinaryClassifier, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent

# NTU25 draw order, same 1-based joint pairs as infer_video.py.
EDGES = [(a - 1, b - 1) for a, b in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
    (24, 25), (25, 12))]


class _Segment:
    """One contiguous tracked-person segment: frame index -> NTU25 pose."""

    def __init__(self, first, min_frames, stride):
        self.first = first
        self.last = first
        self.poses = {}
        self.next_end = first + min_frames - 1
        self.pending = []
        self.evaluated = set()


class LiveWindows:
    """Causal per-track window scoring for a live stream.

    Offline scoring sees the finished track, so every window is evaluated once
    the segment end is known. Live scoring cannot look ahead, so a scheduled
    endpoint is held pending until a later observation proves the segment
    reached it, or until the segment closes. The resulting windows match
    classify_records exactly.
    """

    def __init__(self, classifier, fps, window=64, stride=8, min_frames=16,
                 max_gap=8, min_coverage=.6, threshold=.5,
                 require_full_window=False):
        self.classifier, self.fps = classifier, fps
        self.window, self.stride = window, stride
        self.min_frames, self.max_gap = min_frames, max_gap
        self.min_coverage, self.threshold = min_coverage, threshold
        self.require_full_window = require_full_window
        self.segments = {}
        self.results = []
        self.latest = {}

    def advance(self, frame):
        """Generate scheduled endpoints; a frame without a pose adds no data."""
        for segment in self.segments.values():
            while segment.next_end <= frame:
                segment.pending.append(segment.next_end)
                segment.next_end += self.stride

    def update(self, frame, track_id, pose):
        """Attach a pose to a track and open a new segment after a long gap."""
        segment = self.segments.get(track_id)
        if segment is None or frame - segment.last > self.max_gap + 1:
            if segment is not None:
                self.close(track_id, segment)
            segment = _Segment(frame, self.min_frames, self.stride)
            self.segments[track_id] = segment
            self.latest[track_id] = None  # a new segment must not inherit a label
        segment.poses[frame] = pose
        segment.last = frame
        self.advance(frame)
        # The segment now reaches this frame, so every pending endpoint at or
        # before it is final and can be scored.
        while segment.pending and segment.pending[0] <= frame:
            end = segment.pending.pop(0)
            if end not in segment.evaluated:
                self._score(track_id, segment, end)

    def close(self, track_id, segment):
        """Offline evaluation appends an explicit endpoint at the segment end."""
        self.advance(segment.last)
        while segment.pending and segment.pending[0] <= segment.last:
            end = segment.pending.pop(0)
            if end not in segment.evaluated:
                self._score(track_id, segment, end)
        if (segment.last not in segment.evaluated
                and segment.last >= segment.first + self.min_frames - 1):
            self._score(track_id, segment, segment.last)

    def flush(self):
        for track_id, segment in list(self.segments.items()):
            self.close(track_id, segment)

    def _score(self, track_id, segment, end):
        start = max(segment.first, end - self.window + 1)
        if self.require_full_window and end - start + 1 < self.window:
            return
        observed = sum(f in segment.poses for f in range(start, end + 1))
        coverage = observed / (end - start + 1)
        if observed < self.min_frames or coverage < self.min_coverage:
            return
        poses = np.stack([segment.poses.get(f, np.full((25, 3), np.nan, np.float32))
                           for f in range(start, end + 1)])
        probability = self.classifier.score(poses)
        result = dict(track_id=track_id, start_frame=start, end_frame=end,
                      start_sec=start / self.fps, end_sec=end / self.fps,
                      observed_frames=observed, coverage=coverage,
                      p_fall=probability,
                      predicted_label=int(probability >= self.threshold))
        self.results.append(result)
        self.latest[track_id] = result
        segment.evaluated.add(end)


def open_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open stream: {source}')
    return cap


def draw(frame, people, width, height, time_sec, args):
    """Overlay skeleton, track id and label, matching the file pipeline."""
    for person in people:
        tid = person['track_id']
        score = person['score']
        state = person['state']
        color = (0, 190, 255) if score is None else (
            (30, 30, 240) if score['predicted_label'] else (50, 210, 50))
        points = person['points']
        visibility = person['visibility']
        for a, b in EDGES:
            if min(visibility[a], visibility[b]) >= args.pose_confidence:
                cv2.line(frame, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
        for point, visible in zip(points, visibility):
            if visible >= args.pose_confidence:
                cv2.circle(frame, tuple(point), 4, color, -1, cv2.LINE_AA)
        x, y = np.clip(person['bbox'][:2], [0, 65], [width - 1, height - 1])
        text = f'ID {tid}: {state}'
        if score is not None:
            text += f' p={score["p_fall"]:.3f}'
        cv2.putText(frame, text, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, color, 2, cv2.LINE_AA)
    banner = ' | '.join(f'ID {p["track_id"]}: {p["state"]}' for p in people) or 'NO POSE - no classification'
    cv2.rectangle(frame, (0, 0), (width, 57), (20, 20, 20), -1)
    cv2.putText(frame, f'{time_sec:.2f}s | {banner}', (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'YOLOX -> RTMPose26 -> NTU25 XY, Z=0 -> fall ONNX',
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .48, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stream', required=True, help='RTSP URL or other cv2.VideoCapture source')
    ap.add_argument('--name', default=None, help='Output sub-directory name')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/stream')
    ap.add_argument('--detector-model', type=Path, default=DEFAULT_DETECTOR)
    ap.add_argument('--pose-model', type=Path, default=DEFAULT_POSE)
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--pose-device', choices=['cpu', 'cuda'], default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--det-confidence', type=float, default=.5)
    ap.add_argument('--joint-confidence', type=float, default=.3)
    ap.add_argument('--pose-confidence', type=float, default=.3)
    ap.add_argument('--num-poses', type=int, default=2)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--min-frames', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=8)
    ap.add_argument('--min-coverage', type=float, default=.6)
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--temporal-method', choices=['legacy', 'linear', 'nearest'], default='legacy')
    ap.add_argument('--interpolation-gap', type=int, default=2)
    ap.add_argument('--require-full-window', action='store_true')
    ap.add_argument('--duration', type=float, default=60,
                    help='Seconds to run; 0 runs until interrupted')
    ap.add_argument('--max-lag-frames', type=float, default=2.0,
                    help='Frames of lag tolerated before dropping to stay in sync')
    ap.add_argument('--drop', choices=['live', 'all'], default='live',
                    help='live drops buffered frames to stay at the live edge; all keeps every frame')
    ap.add_argument('--preview', action='store_true',
                    help='Show the annotated stream in a cv2.imshow window')
    ap.add_argument('--preview-scale', type=float, default=.5)
    ap.add_argument('--no-render', action='store_true')
    args = ap.parse_args()
    if not (1 <= args.min_frames <= args.window <= 150 and args.stride > 0 and args.threads > 0
            and args.max_gap >= 0 and args.num_poses > 0 and args.duration >= 0
            and 0 <= args.threshold <= 1 and 0 <= args.pose_confidence <= 1
            and 0 < args.min_coverage <= 1 and 0 <= args.det_confidence <= 1
            and 0 <= args.joint_confidence <= 1 and args.interpolation_gap >= 0
            and args.max_lag_frames >= 0):
        ap.error('Invalid temporal/confidence parameters')
    if not args.no_render and shutil.which('ffmpeg') is None:
        ap.error('ffmpeg required for H.264 output')

    name = args.name or 'stream'
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args.stream)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        cap.release()
        raise RuntimeError('Stream reports no frame size')

    classifier = BinaryClassifier(args.model, args.device, args.threads,
                                  args.temporal_method, args.interpolation_gap)
    estimator = PoseEstimator(args.detector_model, args.pose_model,
                               args.pose_device or args.device, args.threads,
                               args.det_confidence, args.num_poses)
    tracker = PoseTracker(max_gap=args.max_gap)
    windows = LiveWindows(classifier, fps, args.window, args.stride, args.min_frames,
                           args.max_gap, args.min_coverage, args.threshold,
                           args.require_full_window)

    process = None
    if not args.no_render:
        if width % 2 or height % 2:
            width += width % 2
            height += height % 2
        command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                   str(out / 'annotated.mp4')]
        errors = (out / 'ffmpeg.log').open('w')
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)

    started = time.monotonic()
    consumed = processed = skipped = 0
    frame_index = 0
    exhausted = False
    timeline = []
    last_seen, segment_start = {}, {}
    window_name = 'fall-detection'
    try:
        while True:
            if args.duration and time.monotonic() - started >= args.duration:
                break
            # Stay at the live edge: when inference is slower than the camera,
            # drop buffered frames instead of lagging further behind.
            while args.drop == 'live':
                expected = (time.monotonic() - started) * fps
                if consumed + 1 >= expected - args.max_lag_frames:
                    break
                if not cap.grab():
                    exhausted = True
                    break
                consumed += 1
                skipped += 1
                # No pose for this frame, so its window loses the observation.
                windows.advance(consumed)
            if exhausted:
                break
            if not cap.grab():
                break
            consumed += 1
            frame_index = consumed
            ok, frame = cap.retrieve()
            if not ok:
                break
            keypoints, scores, boxes = estimator(frame)
            accepted, centers = [], []
            for xy, confidence, bbox in zip(keypoints, scores, boxes):
                torso = confidence[[5, 6, 11, 12]]
                if (not np.isfinite(xy).all() or not np.isfinite(confidence).all()
                        or float(torso.mean()) < args.pose_confidence):
                    continue
                ntu25, visibility = map_halpe26(xy, confidence, args.joint_confidence)
                if visibility[1] <= 0:
                    continue
                accepted.append(dict(ntu25=ntu25, visibility=visibility, bbox=bbox))
                centers.append(ntu25[0, :2] / [width, height])
            ids = tracker.update_centers(frame_index, centers)
            for tid, record in zip(ids, accepted):
                windows.update(frame_index, tid, record['ntu25'])
            people = []
            for tid, record in zip(ids, accepted):
                if tid not in last_seen or frame_index - last_seen[tid] > args.max_gap + 1:
                    segment_start[tid] = frame_index
                last_seen[tid] = frame_index
                score = windows.latest.get(tid)
                if score is not None and (score['end_frame'] < segment_start[tid]
                                          or frame_index - score['end_frame'] > args.stride + args.max_gap):
                    score = None
                state = 'WARMUP' if score is None else (
                    'FALL' if score['predicted_label'] else 'NON-FALL')
                points = np.rint(np.clip(record['ntu25'][:, :2],
                                         [-width, -height], [2 * width, 2 * height])).astype(int)
                people.append(dict(track_id=tid, state=state, score=score,
                                   p_fall=None if score is None else score['p_fall'],
                                   score_end_frame=None if score is None else score['end_frame'],
                                   points=points, visibility=record['visibility'],
                                   bbox=record['bbox'], time_sec=frame_index / fps))
            processed += 1
            annotated = None
            if process is not None or args.preview:
                pad = cv2.copyMakeBorder(frame, 0, height - frame.shape[0], 0,
                                         width - frame.shape[1], cv2.BORDER_CONSTANT)
                draw(pad, people, width, height, frame_index / fps, args)
                annotated = pad
            if process is not None:
                process.stdin.write(annotated.tobytes())
            if args.preview:
                shown = annotated
                if args.preview_scale != 1:
                    shown = cv2.resize(annotated, (int(width * args.preview_scale),
                                                   int(height * args.preview_scale)),
                                       interpolation=cv2.INTER_AREA)
                cv2.imshow(window_name, shown)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            timeline.append(dict(frame=frame_index, time_sec=frame_index / fps,
                                 people=[dict(track_id=p['track_id'], state=p['state'],
                                              p_fall=p['p_fall'],
                                              score_end_frame=p['score_end_frame'])
                                         for p in people]))
            if processed % 50 == 0:
                rate = processed / max(time.monotonic() - started, 1e-6)
                print(f'{processed} processed, {skipped} dropped, {rate:.1f} fps', flush=True)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        windows.flush()
        if process is not None:
            process.stdin.close()
            process.wait()
        cap.release()

    scores = windows.results
    fields = ['track_id', 'start_frame', 'end_frame', 'start_sec', 'end_sec',
              'observed_frames', 'coverage', 'p_fall', 'predicted_label']
    with (out / 'windows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)
    with (out / 'frames.jsonl').open('w') as f:
        for row in timeline:
            f.write(json.dumps(row) + '\n')
    summary = dict(stream=args.stream, frames=frame_index, fps=fps,
                   processed=processed, dropped=skipped, consumed=consumed,
                   windows=len(scores), fall_windows=sum(r['predicted_label'] for r in scores),
                   max_p_fall=max((r['p_fall'] for r in scores), default=None),
                   provider=classifier.device, pose_providers=estimator.providers,
                   pipeline='yolox_rtmpose_halpe26', adapter=ADAPTER_VERSION,
                   model=str(args.model.resolve()), threshold=args.threshold,
                   first_fall_sec=next((r['end_sec'] for r in scores if r['predicted_label']), None),
                   temporal=dict(method=args.temporal_method, source_window=args.window,
                                 nominal_window_sec=args.window / fps, stride=args.stride,
                                 min_observed_frames=args.min_frames, max_gap=args.max_gap,
                                 interpolation_gap=args.interpolation_gap,
                                 require_full_window=args.require_full_window),
                   elapsed_sec=time.monotonic() - started, ground_truth=None,
                   output=str((out / 'annotated.mp4').resolve()) if not args.no_render else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
