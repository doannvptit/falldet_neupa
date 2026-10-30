"""Video -> YOLOX-s -> RTMPose-m Halpe26 -> NTU25 -> fall ONNX."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import onnxruntime as ort
from fall_onnx.adapter import PoseTracker
from fall_onnx.halpe26 import map_halpe26, ADAPTER_VERSION
from fall_onnx.pose import PoseEstimator, DEFAULT_DETECTOR, DEFAULT_POSE
from fall_onnx.classifier import BinaryClassifier, DEFAULT_MODEL
from fall_onnx.windows import classify_records
ROOT = Path(__file__).resolve().parent

def save_records(path, records):
    shapes = {'halpe26': (26,2), 'scores26': (26,), 'bbox': (4,),
              'ntu25': (25,3), 'visibility25': (25,)}
    arrays = {key: np.stack([r[key] for r in records]) if records
              else np.empty((0, *shape), np.float32) for key, shape in shapes.items()}
    arrays.update(frame=np.array([r['frame'] for r in records], dtype=np.int64),
                  track_id=np.array([r['track_id'] for r in records], dtype=np.int64))
    np.savez_compressed(path, **arrays)


def load_records(path):
    with np.load(path, allow_pickle=False) as data:
        return [{key: (int(data[key][i]) if key in ('frame', 'track_id') else data[key][i].copy())
                 for key in data.files} for i in range(len(data['frame']))]


def extraction_signature(args):
    return dict(schema=2, pipeline='yolox_rtmpose_halpe26', adapter=ADAPTER_VERSION,
        detector_sha256=hashlib.sha256(args.detector_model.read_bytes()).hexdigest(),
        pose_sha256=hashlib.sha256(args.pose_model.read_bytes()).hexdigest(),
        **{k:getattr(args,k) for k in ('max_frames','max_gap','pose_confidence',
                                     'joint_confidence','det_confidence','num_poses')})


def validate_cache(meta, video, signature):
    stat=video.stat()
    if (meta.get('source') != str(video.resolve()) or meta.get('source_size') != stat.st_size
            or meta.get('source_mtime_ns') != stat.st_mtime_ns
            or meta.get('extraction_signature') != signature):
        raise ValueError('Cache source/models/adapter/options mismatch; rerun without --reuse-poses')


def extract(video, args, cv2, estimator):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid FPS')
    meta = dict(fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    tracker = PoseTracker(max_gap=args.max_gap)
    records, frame_index = [], 0
    try:
        while not args.max_frames or frame_index < args.max_frames:
            ok, frame = cap.read()
            if not ok: break
            keypoints, scores, boxes = estimator(frame)  # BGR -> pixel XY
            accepted, centers = [], []
            for xy, confidence, bbox in zip(keypoints,scores,boxes):
                torso=confidence[[5,6,11,12]]
                if (not np.isfinite(xy).all() or not np.isfinite(confidence).all()
                        or float(torso.mean()) < args.pose_confidence):
                    continue
                ntu25, visibility = map_halpe26(xy,confidence,args.joint_confidence)
                # A valid spine center is required by the trained normalizer.
                if visibility[1] <= 0: continue
                accepted.append(dict(halpe26=xy,scores26=confidence,bbox=bbox,
                                     ntu25=ntu25,visibility25=visibility))
                centers.append(ntu25[0,:2]/[meta['width'],meta['height']])
            ids = tracker.update_centers(frame_index,centers)
            for tid, record in zip(ids,accepted):
                records.append(dict(frame=frame_index,track_id=tid,**record))
            frame_index += 1
            if frame_index % 50 == 0:
                print(f'{video.name}: extracted {frame_index} frames, {len(records)} person poses',flush=True)
    finally:
        cap.release()
    if frame_index == 0: raise ValueError('Video contains no decodable frames')
    meta['frames'] = frame_index
    return records,meta


def render(video, output, records, scores, meta, args, cv2):
    from collections import defaultdict
    edges = [(a-1,b-1) for a,b in ((1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(23,8),(24,25),(25,12))]
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
    latest, timeline, last_seen, segment_start = {}, [], {}, {}
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
                    if tid not in last_seen or frame_index-last_seen[tid] > args.max_gap+1:
                        segment_start[tid] = frame_index
                    last_seen[tid] = frame_index
                    score = latest.get(tid)
                    # A previous segment must not leak its prediction into a new track segment.
                    if score is not None and (score['end_frame'] < segment_start[tid]
                            or frame_index-score['end_frame'] > args.stride+args.max_gap):
                        score = None
                    state = 'WARMUP' if score is None else ('FALL' if score['predicted_label'] else 'NON-FALL')
                    color = (0, 190, 255) if score is None else ((30, 30, 240) if score['predicted_label'] else (50, 210, 50))
                    points = record['ntu25'][:, :2]
                    points = np.rint(np.clip(points, [-width, -height], [2*width, 2*height])).astype(int)
                    for a, b in edges:
                        if min(record['visibility25'][a], record['visibility25'][b]) >= args.pose_confidence:
                            cv2.line(frame, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
                    for point, visibility in zip(points, record['visibility25']):
                        if visibility >= args.pose_confidence:
                            cv2.circle(frame, tuple(point), 4, color, -1, cv2.LINE_AA)
                    x, y = np.clip(record['bbox'][:2], [0,65], [width-1,height-1])
                    text = f'ID {tid}: {state}' + (f' p={score["p_fall"]:.3f}' if score else '')
                    cv2.putText(frame, text, (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
                    people.append(dict(track_id=tid, state=state,
                                       p_fall=None if score is None else score['p_fall'],
                                       score_end_frame=None if score is None else score['end_frame']))
                banner = ' | '.join(f'ID {p["track_id"]}: {p["state"]}' for p in people) or 'NO POSE - no classification'
                cv2.rectangle(frame, (0, 0), (width, 57), (20, 20, 20), -1)
                cv2.putText(frame, f'{frame_index/meta["fps"]:.2f}s | {banner}', (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                pipeline_text = 'YOLOX -> RTMPose26 -> NTU25 XY, Z=0 -> fall ONNX'
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
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--video-dir', type=Path)
    ap.add_argument('--output-dir', type=Path, default=ROOT/'outputs/rtmpose26')
    ap.add_argument('--pose-model', type=Path, default=DEFAULT_POSE)
    ap.add_argument('--detector-model', type=Path, default=DEFAULT_DETECTOR)
    ap.add_argument('--pose-device', choices=['cpu','cuda'], default=None, help='Default: same as --device')
    ap.add_argument('--det-confidence', type=float, default=.5)
    ap.add_argument('--joint-confidence', type=float, default=.3)
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--temporal-method',choices=['legacy','linear','nearest'],default='legacy')
    ap.add_argument('--interpolation-gap',type=int,default=2,help='Maximum missing source frames bridged per joint')
    ap.add_argument('--require-full-window',action='store_true',help='Wait for the full source window before scoring')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--min-frames', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=8)
    ap.add_argument('--min-coverage', type=float, default=.6)
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--pose-confidence', type=float, default=.3)
    ap.add_argument('--num-poses', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--reuse-poses', action='store_true')
    ap.add_argument('--no-render', action='store_true',
                    help='Skip the annotated video; keeps windows.csv and summary.json')
    args = ap.parse_args()
    if not (1 <= args.min_frames <= args.window <= 150 and args.stride > 0 and args.threads > 0
            and args.max_gap >= 0 and args.num_poses > 0 and args.max_frames >= 0
            and 0 <= args.threshold <= 1 and 0 <= args.pose_confidence <= 1 and 0 < args.min_coverage <= 1
            and 0 <= args.det_confidence <= 1 and 0 <= args.joint_confidence <= 1
            and args.interpolation_gap>=0):
        ap.error('Invalid temporal/confidence parameters')
    if shutil.which('ffmpeg') is None:
        ap.error('ffmpeg required for H.264 output')
    videos = [args.video] if args.video else sorted(p for p in args.video_dir.iterdir()
        if p.suffix.lower() in ('.mp4','.avi','.mov','.mkv'))
    if not videos or any(not p.is_file() for p in videos):
        ap.error('No valid videos found')
    if len({p.stem for p in videos}) != len(videos):
        ap.error('Video stems must be unique')
    import cv2
    signature=extraction_signature(args)
    classifier = BinaryClassifier(args.model,args.device,args.threads,args.temporal_method,args.interpolation_gap)
    estimator = None if args.reuse_poses else PoseEstimator(args.detector_model,args.pose_model,
        args.pose_device or args.device,args.threads,args.det_confidence,args.num_poses)
    summaries = []
    for video in videos:
        started = time.monotonic()
        out = args.output_dir/video.stem
        if video.resolve() == (out/'annotated.mp4').resolve():
            ap.error('Output would overwrite source video')
        out.mkdir(parents=True, exist_ok=True)
        if args.reuse_poses:
            records = load_records(out/'poses.npz')
            meta = json.loads((out/'extraction.json').read_text())
            validate_cache(meta,video,signature)
        else:
            records, meta = extract(video,args,cv2,estimator)
            meta.update(source=str(video.resolve()),source_size=video.stat().st_size,
                source_mtime_ns=video.stat().st_mtime_ns,pose_model=str(args.pose_model.resolve()),
                detector_model=str(args.detector_model.resolve()),
                pose_coordinates='image2d_multiview',extraction_signature=signature,
                providers=estimator.providers,
                versions=dict(numpy=np.__version__,opencv=cv2.__version__,onnxruntime=ort.__version__))
            save_records(out/'poses.npz', records)
            (out/'extraction.json').write_text(json.dumps(meta,indent=2))
        scores = classify_records(records, classifier, meta['fps'], args.window,
            args.stride, args.min_frames, args.max_gap, args.min_coverage, args.threshold,args.require_full_window)
        fields = ['track_id','start_frame','end_frame','start_sec','end_sec','observed_frames','coverage','p_fall','predicted_label']
        with (out/'windows.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(scores)
        timeline = [] if args.no_render else render(video,out/'annotated.mp4',records,scores,meta,args,cv2)
        with (out/'frames.jsonl').open('w') as f:
            for row in timeline:
                f.write(json.dumps(row)+'\n')
        summary = dict(video=str(video.resolve()), frames=meta['frames'], fps=meta['fps'],
            pose_frames=len({r['frame'] for r in records}), person_poses=len(records),
            windows=len(scores), fall_windows=sum(r['predicted_label'] for r in scores),
            max_p_fall=max((r['p_fall'] for r in scores),default=None),
            provider=classifier.device, pose_providers=meta['providers'], pipeline='yolox_rtmpose_halpe26',
            adapter=ADAPTER_VERSION, model=str(args.model.resolve()), threshold=args.threshold,
            first_fall_sec=next((r['end_sec'] for r in scores if r['predicted_label']),None),
            temporal=dict(method=args.temporal_method,source_window=args.window,
                nominal_window_sec=args.window/meta['fps'],stride=args.stride,min_observed_frames=args.min_frames,
                max_gap=args.max_gap,interpolation_gap=args.interpolation_gap,
                require_full_window=args.require_full_window),
            elapsed_sec=time.monotonic()-started, ground_truth=None,
            output=str((out/'annotated.mp4').resolve()) if not args.no_render else None)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))

if __name__ == '__main__':
    main()
