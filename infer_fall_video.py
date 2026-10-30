"""RGB -> image XY landmarks -> approximate NTU25 -> multiview binary NeuroPath.

Legacy XYZ/PCA checkpoints remain selectable through --config.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/falldet-matplotlib')
import numpy as np

from video_fall.adapter import PoseTracker, map_landmarks, map_visibility

ROOT = Path(__file__).resolve().parent


def save_records(path, records):
    shapes = {'image33': (33, 3), 'world33': (33, 3), 'visibility33': (33,),
              'ntu25': (25, 3), 'visibility25': (25,)}
    arrays = {key: np.stack([r[key] for r in records]) if records
              else np.empty((0, *shape), np.float32) for key, shape in shapes.items()}
    arrays.update(frame=np.array([r['frame'] for r in records], dtype=np.int64),
                  track_id=np.array([r['track_id'] for r in records], dtype=np.int64))
    np.savez_compressed(path, **arrays)


def load_records(path):
    with np.load(path, allow_pickle=False) as data:
        return [{key: (int(data[key][i]) if key in ('frame', 'track_id') else data[key][i].copy())
                 for key in data.files} for i in range(len(data['frame']))]


def extract(video, args, cv2, mp):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid FPS; timestamps cannot be inferred')
    meta = dict(fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.pose_model),
                                        delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=args.num_poses, min_pose_detection_confidence=args.pose_confidence,
        min_pose_presence_confidence=args.pose_confidence,
        min_tracking_confidence=args.pose_confidence)
    tracker = PoseTracker(max_gap=args.max_gap)
    records, frame_index, last_timestamp = [], 0, -1
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
            while not args.max_frames or frame_index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = max(last_timestamp + 1, round(frame_index * 1000 / fps))
                last_timestamp = timestamp
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                accepted = []
                image_only = getattr(args, 'input_coordinate_mode', '') == 'image2d_multiview'
                for person_index, image in enumerate(result.pose_landmarks):
                    image33 = np.array([[p.x, p.y, 0. if image_only else p.z] for p in image], np.float32)
                    world33 = (np.zeros((33, 3), np.float32) if image_only else
                               np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[person_index]], np.float32))
                    visibility = np.array([p.visibility for p in image], np.float32)
                    if (np.isfinite(world33).all() and np.isfinite(image33).all()
                            and float(visibility[[11, 12, 23, 24]].mean()) >= args.pose_confidence):
                        ntu25 = (map_landmarks(image33) * [meta['width'], meta['height'], 0]
                                 if image_only else map_landmarks(world33, camera_axes=True))
                        accepted.append(dict(image33=image33, world33=world33, visibility33=visibility,
                                             ntu25=ntu25,
                                             visibility25=map_visibility(visibility)))
                ids = tracker.update(frame_index, [r['image33'] for r in accepted])
                for tid, record in zip(ids, accepted):
                    records.append(dict(frame=frame_index, track_id=tid, **record))
                frame_index += 1
                if frame_index % 50 == 0:
                    print(f'{video.name}: extracted {frame_index} frames, {len(records)} person poses', flush=True)
    finally:
        cap.release()
    if frame_index == 0:
        raise ValueError('Video contains no decodable frames')
    meta['frames'] = frame_index
    return records, meta


def render(video, output, records, scores, meta, args, cv2):
    from collections import defaultdict
    from graph.ntu_rgb_d import joint_info
    edges = joint_info([25, 10, 5])[1][0]
    by_frame, by_end = defaultdict(list), defaultdict(list)
    for record in records:
        by_frame[record['frame']].append(record)
    for score in scores:
        by_end[score['end_frame']].append(score)
    width, height = meta['width'], meta['height']
    if width % 2 or height % 2:
        width += width % 2
        height += height % 2
    # Pipe RGB annotations to ffmpeg so the result is H.264 and browser playable.
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(meta['fps']), '-i', '-',
               '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    cap = cv2.VideoCapture(str(video))
    latest, timeline = {}, []
    with (output.parent/'ffmpeg.log').open('w') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame_index in range(meta['frames']):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Video decoding stopped unexpectedly at frame {frame_index}')
                frame = cv2.copyMakeBorder(frame, 0, height-meta['height'], 0, width-meta['width'], cv2.BORDER_CONSTANT)
                for score in by_end[frame_index]:
                    latest[score['track_id']] = score
                people = []
                for record in by_frame[frame_index]:
                    tid = record['track_id']
                    score = latest.get(tid)
                    # A previous segment must not leak its prediction into a new track segment.
                    if score is not None and frame_index-score['end_frame'] > args.stride+args.max_gap:
                        score = None
                    state = 'WARMUP' if score is None else ('FALL' if score['predicted_label'] else 'NON-FALL')
                    color = (0, 190, 255) if score is None else ((30, 30, 240) if score['predicted_label'] else (50, 210, 50))
                    points = map_landmarks(record['image33'])[:, :2] * [meta['width'], meta['height']]
                    points = np.rint(np.clip(points, [-width, -height], [2*width, 2*height])).astype(int)
                    for a, b in edges:
                        if min(record['visibility25'][a], record['visibility25'][b]) >= args.pose_confidence:
                            cv2.line(frame, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
                    for point, visibility in zip(points, record['visibility25']):
                        if visibility >= args.pose_confidence:
                            cv2.circle(frame, tuple(point), 4, color, -1, cv2.LINE_AA)
                    x, y = np.clip(points.min(axis=0), [0, 65], [width-1, height-1])
                    text = f'ID {tid}: {state}' + (f' p={score["p_fall"]:.3f}' if score else '')
                    cv2.putText(frame, text, (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
                    people.append(dict(track_id=tid, state=state,
                                       p_fall=None if score is None else score['p_fall'],
                                       score_end_frame=None if score is None else score['end_frame']))
                banner = ' | '.join(f'ID {p["track_id"]}: {p["state"]}' for p in people) or 'NO POSE - no classification'
                cv2.rectangle(frame, (0, 0), (width, 57), (20, 20, 20), -1)
                cv2.putText(frame, f'{frame_index/meta["fps"]:.2f}s | {banner}', (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                mode = getattr(args, 'input_coordinate_mode', '')
                pipeline_text = ('Image XY -> NTU25, Z=0 -> NeuroPath binary (multiview)'
                                 if mode == 'image2d_multiview' else
                                 'MediaPipe -> NTU25 -> PCA XY, Z=0 -> NeuroPath binary'
                                 if mode == 'pca2d_zero_z' else
                                 'MediaPipe 3D estimate -> NTU25 -> NeuroPath binary (experimental RGB adapter)')
                cv2.putText(frame, pipeline_text,
                            (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .48, (200, 200, 200), 1, cv2.LINE_AA)
                process.stdin.write(frame.tobytes())
                timeline.append(dict(frame=frame_index, time_sec=frame_index/meta['fps'], people=people))
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'ffmpeg failed; see {output.parent/"ffmpeg.log"}')
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            cap.release()
    return timeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--video', type=Path)
    source.add_argument('--video-dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=ROOT/'outputs/mediapipe_fall_image2d')
    ap.add_argument('--pose-model', type=Path, default=ROOT/'models/pose_landmarker_full.task')
    ap.add_argument('--config', type=Path, default=ROOT/'config/fall-detection/image2d_multiview_2epoch.yaml')
    ap.add_argument('--weights', type=Path, default=None,
                    help='Default: best checkpoint under the selected config work_dir')
    ap.add_argument('--device', default='auto', help='Classifier: auto, cpu, cuda:0. Pose uses CPU.')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--window', type=int, default=64, help='Source frames per causal window (<=150)')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--min-frames', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=8)
    ap.add_argument('--min-coverage', type=float, default=.6)
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--pose-confidence', type=float, default=.5)
    ap.add_argument('--num-poses', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=0, help='0 processes the entire video')
    ap.add_argument('--reuse-poses', action='store_true', help='Reuse poses.npz and extraction.json in output directory')
    ap.add_argument('--check', action='store_true', help='Check dependencies/assets without running inference')
    args = ap.parse_args()
    if not (1 <= args.min_frames <= args.window <= 150 and args.stride > 0 and args.threads > 0
            and args.max_gap >= 0 and args.num_poses > 0 and args.max_frames >= 0
            and 0 <= args.threshold <= 1 and 0 <= args.pose_confidence <= 1 and 0 < args.min_coverage <= 1):
        ap.error('Invalid temporal, confidence or worker parameters')
    import shutil
    if args.weights is None and args.config.is_file():
        import yaml
        config = yaml.safe_load(args.config.read_text())
        work_dir = Path(config['work_dir'])
        if not work_dir.is_absolute():
            work_dir = ROOT/work_dir
        candidates = list((work_dir/'best').glob('weights-*.pt'))
        if candidates:
            args.weights = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    checks = {name: importlib.util.find_spec(name) is not None for name in ('cv2', 'mediapipe', 'torch')}
    checks.update(pose_model=args.pose_model.is_file(), weights=args.weights is not None and args.weights.is_file(),
                  config=args.config.is_file(), ffmpeg=shutil.which('ffmpeg') is not None)
    required = [k for k, ok in checks.items() if not ok and not (args.reuse_poses and k in ('mediapipe', 'pose_model'))]
    if args.check:
        print(json.dumps(dict(checks=checks, missing=required), indent=2))
        raise SystemExit(bool(required))
    if required:
        ap.error(f'Missing dependencies/assets: {", ".join(required)}. Run scripts/setup_video_fall.sh first.')
    if args.video is None and args.video_dir is None:
        ap.error('Provide --video or --video-dir')
    videos = [args.video] if args.video else sorted(p for p in args.video_dir.iterdir()
                                                   if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))
    if not videos or any(not p.is_file() for p in videos):
        ap.error('No input video files found')
    if len({p.stem for p in videos}) != len(videos):
        ap.error('Video stems must be unique to avoid output collisions')
    import cv2
    from video_fall.classifier import BinaryClassifier, classify_records
    if not args.reuse_poses:
        import mediapipe as mp
    classifier = BinaryClassifier(args.config, args.weights, args.device, args.threads)
    projection_path = None
    if classifier.projection is not None:
        import yaml
        classifier_config = yaml.safe_load(args.config.read_text())
        projection_path = Path(classifier_config['test_feeder_args']['projection_path'])
    args.input_coordinate_mode = 'pca2d_zero_z' if projection_path else classifier.coordinate_mode
    summaries = []
    for video in videos:
        started = time.monotonic()
        out = args.output_dir/video.stem
        if video.resolve() == (out/'annotated.mp4').resolve():
            raise ValueError('Output video would overwrite the input; choose another --output-dir')
        out.mkdir(parents=True, exist_ok=True)
        if args.reuse_poses:
            records = load_records(out/'poses.npz')
            meta = json.loads((out/'extraction.json').read_text())
            if meta['source'] != str(video.resolve()):
                raise ValueError('Cached poses belong to a different video')
            stat = video.stat()
            if (meta['source_size'] != stat.st_size or meta['source_mtime_ns'] != stat.st_mtime_ns
                    or any(meta[k] != getattr(args, k) for k in ('max_frames', 'max_gap', 'pose_confidence', 'num_poses'))):
                raise ValueError('Video or extraction options changed; rerun without --reuse-poses')
        else:
            records, meta = extract(video, args, cv2, mp)
            meta.update(source=str(video.resolve()), pose_model=str(args.pose_model.resolve()),
                        max_gap=args.max_gap, pose_confidence=args.pose_confidence,
                        num_poses=args.num_poses, mediapipe_version=mp.__version__,
                        source_size=video.stat().st_size, source_mtime_ns=video.stat().st_mtime_ns,
                        max_frames=args.max_frames,
                        pose_coordinates=args.input_coordinate_mode,
                        pose_model_sha256=hashlib.sha256(args.pose_model.read_bytes()).hexdigest())
            save_records(out/'poses.npz', records)
            (out/'extraction.json').write_text(json.dumps(meta, indent=2))
        print(f'{video.name}: classify {len(records)} poses on {classifier.device}', flush=True)
        classification_records = records
        if args.input_coordinate_mode == 'image2d_multiview':
            classification_records = [dict(r, ntu25=map_landmarks(r['image33']) *
                                          [meta['width'], meta['height'], 0]) for r in records]
        elif meta.get('pose_coordinates') == 'image2d_multiview':
            raise ValueError('Image-only cache has no world coordinates; extract again for XYZ/PCA models')
        scores = classify_records(classification_records, classifier, meta['fps'], args.window, args.stride,
                                  args.min_frames, args.max_gap, args.min_coverage, args.threshold)
        fields = ['track_id', 'start_frame', 'end_frame', 'start_sec', 'end_sec',
                  'observed_frames', 'coverage', 'p_fall', 'predicted_label']
        with (out/'windows.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(scores)
        timeline = render(video, out/'annotated.mp4', records, scores, meta, args, cv2)
        with (out/'frames.jsonl').open('w') as f:
            for row in timeline:
                f.write(json.dumps(row)+'\n')
        summary = dict(video=str(video.resolve()), **{k: meta[k] for k in ('frames', 'fps')},
                       pose_frames=len({r['frame'] for r in records}), person_poses=len(records),
                       track_count=len({r['track_id'] for r in records}), windows=len(scores),
                       fall_windows=sum(r['predicted_label'] for r in scores),
                       threshold=args.threshold, device=str(classifier.device),
                       pose_model=meta['pose_model'], mediapipe_version=meta['mediapipe_version'],
                       weights=str(args.weights.resolve()), config=str(args.config.resolve()),
                       input_coordinate_mode=args.input_coordinate_mode,
                       projection_path=str(projection_path.resolve()) if projection_path else None,
                       projection_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest() if projection_path else None,
                       parameters={k: getattr(args, k) for k in ('window', 'stride', 'min_frames', 'max_gap', 'min_coverage')},
                       elapsed_sec=time.monotonic()-started, ground_truth=None,
                       limitation=('2D image landmarks + approximate NTU mapping; synthetic camera training, not RGB-validated accuracy'
                                   if args.input_coordinate_mode == 'image2d_multiview' else
                                   'Estimated hip-relative 3D + approximate NTU mapping; not RGB-validated accuracy'),
                       output=str((out/'annotated.mp4').resolve()))
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
