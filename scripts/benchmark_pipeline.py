"""Performance benchmark for the ONNX fall pipeline: per stage and end to end.

Measures the real video path (decode, YOLOX, RTMPose, Halpe26->NTU25 mapping,
tracking, windowing/preprocessing/classification, overlay encoding) on the
sample videos, plus micro-benchmarks per ONNX model and CPU thread scaling.
Only NumPy, OpenCV and ONNX Runtime are required.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_onnx.adapter import PoseTracker
from fall_onnx.classifier import BinaryClassifier, DEFAULT_MODEL
from fall_onnx.halpe26 import map_halpe26
from fall_onnx.pose import PoseEstimator, DEFAULT_DETECTOR, DEFAULT_POSE
from fall_onnx.preprocess import image_model_input
from fall_onnx.temporal import temporal_model_input
from fall_onnx.windows import classify_records
from infer_video import render


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def shell(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        return None


def environment():
    """Hardware and software facts that the numbers depend on."""
    info = dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                numpy=np.__version__, opencv=cv2.__version__,
                onnxruntime=ort.__version__, providers=ort.get_available_providers(),
                ffmpeg=shell('ffmpeg -version'))
    cpu = shell("lscpu | grep 'Model name' | cut -d: -f2-")
    if cpu:
        info['cpu_model'] = cpu.strip()
    info['physical_cores'] = int(shell("lscpu | grep '^Core(s) per socket' | awk '{print $4}'") or 0)
    info['threads_per_core'] = int(shell("lscpu | grep '^Thread(s) per core' | awk '{print $4}'") or 0)
    info['ram_gb'] = round(int(shell("free -g | awk 'NR==2{print $2}'") or 0))
    gpu = shell("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader")
    if gpu:
        info['gpu'] = gpu
        info['driver_version'] = gpu.split(',')[1].strip()
        info['cuda_driver'] = shell('nvidia-smi | grep -o "CUDA Version: [0-9.]*"')
    return info


def model_specs(detector, pose, model):
    """Static facts of the three ONNX models that shape the runtime cost."""
    specs = []
    for name, path in (('yolox_s', detector), ('rtmpose_m_halpe26', pose), ('fall_image2d', model)):
        path = Path(path)
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        entry = dict(name=name, file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                     sha256=sha256(path), input=[(i.name, i.shape, i.type) for i in session.get_inputs()],
                     output=[(o.name, o.shape, o.type) for o in session.get_outputs()])
        if name == 'fall_image2d':
            report = json.loads(path.with_suffix('.json').read_text())
            entry.update(opset=report.get('opset'), parameters=report.get('parameters'),
                         checkpoint=Path(report['checkpoint']).name,
                         checkpoint_sha256=report.get('checkpoint_sha256'),
                         export_check=report.get('verification'))
        specs.append(entry)
    return specs


def timing(function, repeat, warmup=3):
    """Return latency statistics in milliseconds for a callable."""
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    array = np.asarray(samples)
    return dict(calls=repeat, mean_ms=round(float(array.mean()), 3),
                median_ms=round(float(np.median(array)), 3),
                p95_ms=round(float(np.percentile(array, 95)), 3),
                min_ms=round(float(array.min()), 3), max_ms=round(float(array.max()), 3))


def run_pipeline(video, estimator, classifier, args, out_dir):
    """One full pass with per-stage timers, identical to the production path."""
    timers = defaultdict(float)
    counts = defaultdict(int)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = PoseTracker(max_gap=args.max_gap)
    records = []
    started = time.perf_counter()
    frame_index = 0
    while True:
        mark = time.perf_counter()
        ok, frame = cap.read()
        timers['decode'] += time.perf_counter() - mark
        if not ok:
            break
        mark = time.perf_counter()
        boxes = np.asarray(estimator.detector(frame), np.float32).reshape(-1, 4)
        timers['detect'] += time.perf_counter() - mark
        mark = time.perf_counter()
        if len(boxes):
            keypoints, scores = estimator.pose(frame, bboxes=boxes)
        else:
            keypoints = np.empty((0, 26, 2), np.float32)
            scores = np.empty((0, 26), np.float32)
        timers['pose'] += time.perf_counter() - mark
        counts['pose_calls'] += len(boxes)
        mark = time.perf_counter()
        accepted, centers = [], []
        for xy, confidence, bbox in zip(keypoints, scores, boxes):
            torso = confidence[[5, 6, 11, 12]]
            if (not np.isfinite(xy).all() or not np.isfinite(confidence).all()
                    or float(torso.mean()) < args.pose_confidence):
                continue
            ntu25, visibility = map_halpe26(xy, confidence, args.joint_confidence)
            if visibility[1] <= 0:
                continue
            accepted.append(dict(halpe26=xy, scores26=confidence, bbox=bbox,
                                 ntu25=ntu25, visibility25=visibility))
            centers.append(ntu25[0, :2] / [width, height])
        ids = tracker.update_centers(frame_index, centers)
        timers['map_track'] += time.perf_counter() - mark
        for tid, record in zip(ids, accepted):
            records.append(dict(frame=frame_index, track_id=tid, **record))
        frame_index += 1
    cap.release()
    extract_total = time.perf_counter() - started

    mark = time.perf_counter()
    scores = classify_records(records, classifier, fps, args.window, args.stride,
                               args.min_frames, args.max_gap, args.min_coverage,
                               args.threshold, args.require_full_window)
    timers['classify'] += time.perf_counter() - mark
    counts['windows'] = len(scores)

    meta = dict(fps=fps, width=width, height=height, frames=frame_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_args = argparse.Namespace(stride=args.stride, max_gap=args.max_gap,
                                     pose_confidence=args.pose_confidence)
    mark = time.perf_counter()
    render(video, out_dir/'annotated.mp4', records, scores, meta, render_args, cv2)
    timers['render'] += time.perf_counter() - mark

    total = time.perf_counter() - started
    stages = {k: round(v * 1000, 2) for k, v in timers.items()}
    accounted = sum(timers.values())
    return dict(video=video.name, frames=frame_index, fps=fps,
                resolution=f'{width}x{height}',
                person_poses=len(records), tracks=len({r['track_id'] for r in records}),
                windows=len(scores), fall_windows=sum(s['predicted_label'] for s in scores),
                stage_ms=stages, stage_share_pct={k: round(100 * v / accounted, 2)
                                                  for k, v in timers.items()},
                extract_ms=round(extract_total * 1000, 2),
                total_ms=round(total * 1000, 2),
                other_ms=round((total - accounted) * 1000, 2),
                throughput_fps=round(frame_index / total, 2),
                video_seconds=round(frame_index / fps, 3),
                slowdown_vs_realtime=round(total / (frame_index / fps), 2),
                realtime_factor=round((frame_index / fps) / total, 3),
                counts=dict(counts))


def microbenchmarks(video, estimator, classifier, args, repeat):
    """Per-component latency on real data."""
    cap = cv2.VideoCapture(str(video))
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    boxes = np.asarray(estimator.detector(frame), np.float32).reshape(-1, 4)
    if len(boxes):
        keypoints, scores = estimator.pose(frame, bboxes=boxes[:1])
    else:
        keypoints = np.zeros((1, 26, 2), np.float32)
        scores = np.zeros((1, 26), np.float32)
    window = synthetic_window(keypoints[0], scores[0], args.joint_confidence)
    tensor = image_model_input(window)
    tracker = PoseTracker(max_gap=args.max_gap)
    center = np.zeros(2, np.float32)
    results = {}
    results['decode_frame'] = decode_profile(video)
    results['yolox_detect_frame'] = timing(lambda: estimator.detector(frame), repeat)
    results['rtmpose_per_person'] = timing(lambda: estimator.pose(frame, bboxes=boxes[:1]), repeat)
    results['map_halpe26'] = timing(lambda: map_halpe26(keypoints[0], scores[0], args.joint_confidence),
                                    max(repeat * 10, 100))
    results['tracker_update'] = timing(lambda: tracker.update_centers(0, [center]),
                                       max(repeat * 10, 100))
    results['preprocess_legacy'] = timing(lambda: image_model_input(window), repeat)
    results['preprocess_linear'] = timing(lambda: temporal_model_input(window, 'linear', 2), repeat)
    results['classify_onnx'] = timing(lambda: classifier.logits(tensor[None]), max(repeat * 2, 20))
    results['classify_total'] = timing(lambda: classifier.score(window), repeat)
    results.update(detector_breakdown(estimator, frame, repeat))
    results.update(pose_breakdown(estimator, frame, boxes, repeat))
    return results


def detector_breakdown(estimator, frame, repeat):
    """Split YOLOX into letterbox resize, ONNX run and NMS postprocess."""
    detector = estimator.detector
    padded, ratio = detector.preprocess(frame)
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    name = detector.session.get_inputs()[0].name
    output = detector.session.run(None, {name: tensor})[0]
    return dict(detect_preprocess_letterbox=timing(lambda: detector.preprocess(frame), repeat),
                detect_onnx=timing(lambda: detector.session.run(None, {name: tensor}), repeat),
                detect_postprocess_nms=timing(lambda: detector.postprocess(output, ratio), repeat),
                detect_input_bytes=tensor.nbytes)


def pose_breakdown(estimator, frame, boxes, repeat):
    """Split RTMPose into affine crop, ONNX run and simcc decode."""
    if not len(boxes):
        return {}
    pose = estimator.pose
    bbox = boxes[:1]
    image, center, scale = pose.preprocess(frame, bbox[0])
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
    name = pose.session.get_inputs()[0].name
    outputs = pose.session.run(None, {name: tensor})
    return dict(pose_preprocess_affine=timing(lambda: pose.preprocess(frame, bbox[0]), repeat),
                pose_onnx=timing(lambda: pose.session.run(None, {name: tensor}), repeat),
                pose_postprocess_simcc=timing(lambda: pose.postprocess(outputs, center, scale), repeat),
                pose_input_bytes=tensor.nbytes)


def _decode_once(video):
    """Decode one frame of a kept-open video (no reopen cost)."""
    cap = cv2.VideoCapture(str(video))
    cap.read()
    cap.release()


def decode_profile(video, max_frames=120):
    """Sequential decode latency of a real video, excluding file open cost."""
    cap = cv2.VideoCapture(str(video))
    samples = []
    while len(samples) < max_frames:
        mark = time.perf_counter()
        ok, _ = cap.read()
        elapsed = (time.perf_counter() - mark) * 1000
        if not ok:
            break
        samples.append(elapsed)
    cap.release()
    array = np.asarray(samples)
    return dict(calls=len(samples), mean_ms=round(float(array.mean()), 3),
                median_ms=round(float(np.median(array)), 3),
                p95_ms=round(float(np.percentile(array, 95)), 3),
                min_ms=round(float(array.min()), 3), max_ms=round(float(array.max()), 3))


def _tracker_step():
    tracker = PoseTracker(max_gap=8)
    tracker.update_centers(0, [np.zeros(2, np.float32)])


def synthetic_window(keypoints, scores, joint_confidence, frames=64):
    """A realistic (T,25,2) NTU25 pixel window: one tracked person with motion."""
    ntu25, _ = map_halpe26(keypoints, scores, joint_confidence)
    base = ntu25[:, :2].astype(np.float32)
    phase = np.linspace(0, 2*np.pi, frames, dtype=np.float32)[:, None, None]
    return base[None] + 2.0*np.sin(phase)*np.array([1., .5], np.float32)[None, None, :]


def thread_scaling(args, frame, boxes, tensor):
    """Latency of every ONNX model versus intra-op thread count (CPU)."""
    scaling = {}
    for threads in args.thread_list:
        detector = PoseEstimator(DEFAULT_DETECTOR, DEFAULT_POSE, 'cpu', threads,
                                 args.det_confidence, args.num_poses)
        classifier = BinaryClassifier(DEFAULT_MODEL, 'cpu', threads)
        entry = dict(yolox_detect_frame=timing(lambda: detector.detector(frame), 10)['median_ms'],
                     rtmpose_per_person=timing(lambda: detector.pose(frame, bboxes=boxes[:1]), 10)['median_ms'],
                     classify_onnx=timing(lambda: classifier.logits(tensor[None]), 30)['median_ms'])
        scaling[str(threads)] = entry
    return scaling


def cold_start_probe(video, device, threads):
    """Measure process start, session creation and first inference latency."""
    start = time.perf_counter()
    from fall_onnx.pose import PoseEstimator as _PoseEstimator
    from fall_onnx.classifier import BinaryClassifier as _Classifier
    imported = time.perf_counter() - start
    mark = time.perf_counter()
    estimator = _PoseEstimator(DEFAULT_DETECTOR, DEFAULT_POSE, device, threads, .5, 2)
    classifier = _Classifier(DEFAULT_MODEL, device, threads)
    sessions = time.perf_counter() - mark
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot decode {video}')
    mark = time.perf_counter()
    boxes = np.asarray(estimator.detector(frame), np.float32).reshape(-1, 4)
    detect_first = time.perf_counter() - mark
    mark = time.perf_counter()
    estimator.detector(frame)
    detect_second = time.perf_counter() - mark
    mark = time.perf_counter()
    if len(boxes):
        estimator.pose(frame, bboxes=boxes[:1])
    pose_first = time.perf_counter() - mark
    mark = time.perf_counter()
    if len(boxes):
        estimator.pose(frame, bboxes=boxes[:1])
    pose_second = time.perf_counter() - mark
    window = np.zeros((64, 25, 2), np.float32)
    window[:, :, 0] = np.linspace(0, 100, 25, dtype=np.float32)
    window[:, :, 1] = np.linspace(0, 200, 25, dtype=np.float32)
    tensor = image_model_input(window)
    mark = time.perf_counter()
    classifier.logits(tensor[None])
    classify_first = time.perf_counter() - mark
    mark = time.perf_counter()
    classifier.logits(tensor[None])
    classify_second = time.perf_counter() - mark
    return dict(device=device, threads=threads,
                import_ms=round(imported * 1000, 2),
                session_create_ms=round(sessions * 1000, 2),
                detect_first_ms=round(detect_first * 1000, 2),
                detect_second_ms=round(detect_second * 1000, 2),
                pose_first_ms=round(pose_first * 1000, 2),
                pose_second_ms=round(pose_second * 1000, 2),
                classify_first_ms=round(classify_first * 1000, 2),
                classify_second_ms=round(classify_second * 1000, 2),
                first_result_ms=round((detect_first + pose_first + classify_first) * 1000, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video-dir', type=Path, default=ROOT.parent/'PdPdFall-ONNX/videos')
    ap.add_argument('--output-dir', type=Path, default=ROOT/'outputs/benchmark')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--repeat', type=int, default=20, help='Micro-benchmark repetitions')
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--min-frames', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=8)
    ap.add_argument('--min-coverage', type=float, default=.6)
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--pose-confidence', type=float, default=.3)
    ap.add_argument('--joint-confidence', type=float, default=.3)
    ap.add_argument('--det-confidence', type=float, default=.5)
    ap.add_argument('--num-poses', type=int, default=2)
    ap.add_argument('--temporal-method', choices=['legacy', 'linear', 'nearest'], default='legacy')
    ap.add_argument('--interpolation-gap', type=int, default=2)
    ap.add_argument('--require-full-window', action='store_true')
    ap.add_argument('--thread-list', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--skip-scaling', action='store_true')
    ap.add_argument('--probe-device', choices=['cpu', 'cuda'],
                    help='Run only the cold-start probe in this process and print JSON')
    ap.add_argument('--micro-only', action='store_true',
                    help='Skip the video pipeline; only micro-benchmarks and cold start')
    args = ap.parse_args()
    if args.repeat < 1 or args.threads < 1:
        ap.error('Invalid repeat/threads')
    if args.probe_device:
        video = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() == '.mp4')[0]
        print('PROBE_JSON ' + json.dumps(cold_start_probe(video, args.probe_device, args.threads)))
        return

    videos = sorted(p for p in args.video_dir.iterdir()
                    if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))
    if not videos:
        ap.error('No input videos found')
    out_root = args.output_dir/args.device
    out_root.mkdir(parents=True, exist_ok=True)

    estimator = PoseEstimator(DEFAULT_DETECTOR, DEFAULT_POSE, args.device, args.threads,
                              args.det_confidence, args.num_poses)
    classifier = BinaryClassifier(DEFAULT_MODEL, args.device, args.threads,
                                  args.temporal_method, args.interpolation_gap)
    report = dict(device=args.device, threads=args.threads,
                  temporal_method=args.temporal_method,
                  environment=environment(),
                  models=model_specs(DEFAULT_DETECTOR, DEFAULT_POSE, DEFAULT_MODEL),
                  providers=dict(detector=estimator.detector.session.get_providers()[0],
                                 pose=estimator.pose.session.get_providers()[0],
                                 classifier=classifier.device),
                  videos=[], micro=None, thread_scaling=None,
                  peak_rss_mb=None)

    for video in ([] if args.micro_only else videos):
        result = run_pipeline(video, estimator, classifier, args, out_root/video.stem)
        report['videos'].append(result)
        print(json.dumps(result), flush=True)

    report['micro'] = microbenchmarks(videos[0], estimator, classifier, args, args.repeat)
    if args.device == 'cpu' and not args.skip_scaling:
        cap = cv2.VideoCapture(str(videos[0]))
        ok, frame = cap.read()
        cap.release()
        detector = PoseEstimator(DEFAULT_DETECTOR, DEFAULT_POSE, 'cpu', args.threads,
                                 args.det_confidence, args.num_poses)
        boxes = np.asarray(detector.detector(frame), np.float32).reshape(-1, 4)
        keypoints, pose_scores = detector.pose(frame, bboxes=boxes[:1])
        window = synthetic_window(keypoints[0], pose_scores[0], args.joint_confidence)
        report['thread_scaling'] = thread_scaling(args, frame, boxes, image_model_input(window))
    report['peak_rss_mb'] = round(peak_rss_mb(), 1)

    cold = {}
    for device in ('cpu', 'cuda'):
        probe = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                '--probe-device', device, '--video-dir', str(args.video_dir),
                                '--threads', str(args.threads)],
                               capture_output=True, text=True, cwd=str(ROOT))
        for line in probe.stdout.splitlines():
            if line.startswith('PROBE_JSON '):
                cold[device] = json.loads(line[len('PROBE_JSON '):])
    report['cold_start'] = cold

    totals = defaultdict(float)
    frames = 0
    for entry in report['videos']:
        for stage, value in entry['stage_ms'].items():
            totals[stage] += value
        frames += entry['frames']
    report['aggregate'] = dict(videos=len(report['videos']), frames=frames,
                                total_stage_ms={k: round(v, 2) for k, v in totals.items()},
                                total_ms=round(sum(e['total_ms'] for e in report['videos']), 2),
                                mean_throughput_fps=round(sum(e['throughput_fps'] for e in report['videos'])
                                                         / max(len(report['videos']), 1), 2))
    path = out_root/('micro.json' if args.micro_only else 'results.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Results: {path}', flush=True)


if __name__ == '__main__':
    main()
