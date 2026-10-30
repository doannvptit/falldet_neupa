"""Controlled 8-FPS temporal-window experiment, no model retraining.

Downsample actual RGB frames before YOLOX/RTMPose. Same pose extraction reused
for all 8-FPS temporal policies. Preserve original playback speed/timestamps.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infer_video import extract,save_records,load_records,render,extraction_signature,validate_cache
from fall_onnx.pose import PoseEstimator,DEFAULT_DETECTOR,DEFAULT_POSE
from fall_onnx.classifier import BinaryClassifier,DEFAULT_MODEL
from fall_onnx.windows import classify_records

POLICIES={
 'legacy_64frames':dict(window=64,stride=8,min_frames=16,method='legacy',full=False),
 'nearest_4s':dict(window=32,stride=2,min_frames=8,method='nearest',full=False),
 'linear_2s':dict(window=16,stride=2,min_frames=8,method='linear',full=False),
 'linear_4s':dict(window=32,stride=2,min_frames=8,method='linear',full=False),
 'linear_8s':dict(window=64,stride=2,min_frames=8,method='linear',full=False),
 'linear_8s_full':dict(window=64,stride=2,min_frames=8,method='linear',full=True),
}


def downsample(source,destination,fps=8.):
    cap=cv2.VideoCapture(str(source))
    source_fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(3)),int(cap.get(4))
    if not cap.isOpened() or source_fps<=0 or total<=0: raise ValueError(f'Invalid video: {source}')
    indices=np.minimum(np.floor(np.arange(math.ceil(total/source_fps*fps))*source_fps/fps+1e-9).astype(int),total-1)
    destination.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264rgb','-crf','0','-preset','veryfast',str(destination)]
    hashes=[];index=0
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for target,source_index in enumerate(indices):
            while index<=source_index:
                ok,frame=cap.read()
                if not ok: raise RuntimeError('Unexpected source EOF')
                index+=1
            process.stdin.write(frame.tobytes())
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        process.stdin.close()
        error=process.stderr.read().decode()
        if process.wait(): raise RuntimeError(error)
    except BaseException:
        process.kill();process.wait();raise
    finally:
        cap.release()
    # Ensure lossless transcoding does not change pixels seen by pose models.
    output=cv2.VideoCapture(str(destination))
    for expected in hashes:
        ok,frame=output.read()
        assert ok and hashlib.sha256(frame.tobytes()).hexdigest()==expected
    ok,_=output.read();assert not ok
    assert abs(output.get(cv2.CAP_PROP_FPS)-fps)<1e-6
    output.release()
    manifest=dict(source=str(source.resolve()),source_fps=source_fps,source_frames=total,
        target_fps=fps,output_frames=len(indices),source_duration=total/source_fps,
        output_duration=len(indices)/fps,pixels_unchanged=True,selection='floor(n * source_fps / target_fps)',
        samples=[dict(frame=n,time_sec=n/fps,source_frame=int(i),source_time_sec=int(i)/source_fps)
                 for n,i in enumerate(indices)])
    destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video-dir',type=Path,default=ROOT.parent/'PdPdFall-ONNX/videos')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/eval_8fps')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--reuse',action='store_true')
    ap.add_argument('--render',action='store_true',help='Render 8-FPS overlays for each policy')
    args=ap.parse_args()
    videos=sorted(p for p in args.video_dir.iterdir() if p.suffix.lower()=='.mp4')
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    extraction_args=SimpleNamespace(max_frames=0,max_gap=2,pose_confidence=.3,joint_confidence=.3,
        det_confidence=.5,num_poses=2,pose_model=DEFAULT_POSE,detector_model=DEFAULT_DETECTOR)
    signature=extraction_signature(extraction_args)
    estimator=None if args.reuse else PoseEstimator(device=args.device,threads=4)
    classifiers={method:BinaryClassifier(device=args.device,temporal_method=method,interpolation_gap=2)
                 for method in ('legacy','nearest','linear')}
    results=[]
    for source in videos:
        sampled=root/'videos_8fps'/source.name
        if not args.reuse:
            sampling=downsample(source,sampled)
        else:
            sampling=json.loads(sampled.with_suffix('.json').read_text())
        cache=root/'poses_8fps'/source.stem;cache.mkdir(parents=True,exist_ok=True)
        if args.reuse:
            meta=json.loads((cache/'extraction.json').read_text())
            validate_cache(meta,sampled,signature)
            records=load_records(cache/'poses.npz')
        else:
            records,meta=extract(sampled,extraction_args,cv2,estimator)
            meta.update(source=str(sampled.resolve()),source_size=sampled.stat().st_size,
                source_mtime_ns=sampled.stat().st_mtime_ns,extraction_signature=signature,
                providers=estimator.providers)
            save_records(cache/'poses.npz',records)
            (cache/'extraction.json').write_text(json.dumps(meta,indent=2))
        baseline=json.loads((ROOT/'outputs/rtmpose26_gpu'/source.stem/'summary.json').read_text())
        result=dict(video=source.stem,sampling=sampling,frames=meta['frames'],fps=meta['fps'],
            pose_frames=len({r['frame'] for r in records}),person_poses=len(records),
            baseline_30fps={k:baseline[k] for k in ('windows','fall_windows','first_fall_sec','max_p_fall')},policies={})
        for name,policy in POLICIES.items():
            out=root/name/source.stem;out.mkdir(parents=True,exist_ok=True)
            scores=classify_records(records,classifiers[policy['method']],8.,policy['window'],policy['stride'],
                policy['min_frames'],2,.6,.5,require_full_window=policy['full'])
            fields=['track_id','start_frame','end_frame','start_sec','end_sec','observed_frames','coverage','p_fall','predicted_label']
            with (out/'windows.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(scores)
            summary=dict(**policy,nominal_window_sec=policy['window']/8,windows=len(scores),
                fall_windows=sum(s['predicted_label'] for s in scores),
                first_fall_sec=next((s['end_sec'] for s in scores if s['predicted_label']),None),
                max_p_fall=max((s['p_fall'] for s in scores),default=None),
                max_actual_span_sec=max(((s['end_frame']-s['start_frame'])/8 for s in scores),default=0),
                min_observed_frames=min((s['observed_frames'] for s in scores),default=0),
                max_observed_frames=max((s['observed_frames'] for s in scores),default=0),
                synthetic_poses_are_observations=False,decision_available_at='window end source frame',
                status='evaluated' if scores else 'insufficient usable history')
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            if args.render:
                render_args=SimpleNamespace(stride=policy['stride'],max_gap=2,pose_confidence=.3)
                timeline=render(sampled,out/'annotated.mp4',records,scores,meta,render_args,cv2)
                with (out/'frames.jsonl').open('w') as f:
                    for row in timeline:f.write(json.dumps(row)+'\n')
            result['policies'][name]=summary
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='sampling'}),flush=True)
        (root/'results.json').write_text(json.dumps(dict(device=args.device,threshold=.5,results=results,
            ground_truth=None,limitation='No accuracy/recall metric without fall interval labels; compare alert behavior and delay only.'),indent=2)+'\n')
    print('Results:',root/'results.json',flush=True)

if __name__=='__main__':main()
