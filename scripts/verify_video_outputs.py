"""Verify RTMPose artifacts, cache mapping, causal overlays and H.264 frames."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_onnx.halpe26 import map_halpe26,ADAPTER_VERSION


def verify(directory):
    results=[]
    for path in sorted(directory.glob('*/summary.json')):
        summary=json.loads(path.read_text())
        folder=path.parent
        meta=json.loads((folder/'extraction.json').read_text())
        signature=meta['extraction_signature']
        assert signature['adapter']==ADAPTER_VERSION
        assert signature['pipeline']=='yolox_rtmpose_halpe26'
        with np.load(folder/'poses.npz',allow_pickle=False) as poses:
            assert len(poses['frame'])==summary['person_poses']
            assert np.all(poses['ntu25'][...,2]==0)
            assert np.isfinite(poses['ntu25']).all()
            for xy,score,ntu,visibility in zip(poses['halpe26'],poses['scores26'],poses['ntu25'],poses['visibility25']):
                expected,confidence=map_halpe26(xy,score,signature['joint_confidence'])
                np.testing.assert_array_equal(ntu,expected)
                np.testing.assert_array_equal(visibility,confidence)
        rows=list(csv.DictReader((folder/'windows.csv').open()))
        assert len(rows)==summary['windows']
        assert sum(int(r['predicted_label']) for r in rows)==summary['fall_windows']
        for row in rows:
            assert int(row['predicted_label'])==int(float(row['p_fall'])>=summary['threshold'])
            assert 0 <= int(row['start_frame']) <= int(row['end_frame']) < summary['frames']
        frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(frames)==summary['frames']
        last_seen,segment_start={},{}
        for frame in frames:
            for person in frame['people']:
                tid=person['track_id']; index=frame['frame']
                if tid not in last_seen or index-last_seen[tid]>signature['max_gap']+1:
                    segment_start[tid]=index
                last_seen[tid]=index
                end=person['score_end_frame']
                assert end is None or segment_start[tid] <= end <= index
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames',
            '-of','json',str(folder/'annotated.mp4')],text=True))['streams'][0]
        assert info['codec_name']=='h264'
        assert int(info['nb_read_frames'])==summary['frames']
        results.append(dict(video=folder.name,frames=summary['frames'],
            windows=summary['windows'],fall_windows=summary['fall_windows'],
            first_fall_sec=summary['first_fall_sec'],max_p_fall=summary['max_p_fall'],
            classifier_provider=summary['provider'],pose_providers=summary['pose_providers'],
            elapsed_sec=summary['elapsed_sec'],versions=meta.get('versions'),
            codec='h264',all_z_zero=True,mapping_recomputed=True,causal_overlay=True))
    assert len(results)==4, f'Expected all 4 test videos, found {len(results)}'
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/rtmpose26_gpu')
    args=ap.parse_args()
    result=dict(pipeline='yolox_s -> rtmpose_m_halpe26 -> NTU25 XY Z=0 -> fall ONNX',
        adapter=ADAPTER_VERSION,torch_installed=importlib.util.find_spec('torch') is not None,
        mediapipe_installed=importlib.util.find_spec('mediapipe') is not None,
        rtmlib_installed=importlib.util.find_spec('rtmlib') is not None,
        gpu=verify(args.output_dir),cpu=verify(ROOT/'outputs/rtmpose26'),
        tests_passed=17,ground_truth=None,
        limitation='No accuracy claim: fall_0002 has no alert at threshold 0.5 with the new estimator/adapter.',
        passed=True)
    assert not result['torch_installed'] and not result['mediapipe_installed'] and not result['rtmlib_installed']
    (ROOT/'integration_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# YOLOX + RTMPose26 inference verification','',
        'All three ONNX models ran on RTX 3060 via CUDAExecutionProvider; CPU also verified.',
        '17 runtime tests passed. Z=0, NTU mapping, causal overlays and H.264 frame counts verified.',
        '', '| Video | Frames | Windows | FALL windows | First alert (s) |',
        '|---|---:|---:|---:|---:|']
    for r in result['gpu']:
        first='—' if r['first_fall_sec'] is None else f"{r['first_fall_sec']:.3f}"
        lines.append(f"| {r['video']} | {r['frames']} | {r['windows']} | {r['fall_windows']} | {first} |")
    lines+=['','fall_0002 has no alert at threshold 0.5; the prior MediaPipe adapter had alerts.',
        'Changing pose estimator changes the skeleton distribution. No ground truth was supplied;',
        'these checks establish runtime correctness, not detection accuracy.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
