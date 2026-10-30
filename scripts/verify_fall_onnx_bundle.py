"""Development verification against training code; not needed by standalone runtime.

Run from the poc directory with its training Python environment.
"""
import csv
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from video_fall.classifier import BinaryClassifier as TorchClassifier, image_model_input as reference
ROOT=Path.cwd().parent/'poc_inference_onnx'
sys.path.insert(0,str(ROOT))
from fall_onnx.preprocess import image_model_input
from fall_onnx.classifier import BinaryClassifier
from fall_onnx.windows import classify_records
from fall_onnx.adapter import map_landmarks
spec=importlib.util.spec_from_file_location('standalone_video',ROOT/'infer_video.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
rng=np.random.default_rng(43)
pre_errors=[]
for length in [1,2,15,16,39,40,54,64,65,100,150,160]:
    p=rng.uniform(1,1000,(length,25,3)).astype(np.float32)
    p[...,2]=np.inf
    for gaps in [False,True]:
        q=p.copy()
        if gaps and length>15:
            q[:2]=np.nan; q[7:10]=0;q[-2:]=np.nan
        expected=reference(q)
        actual=image_model_input(q)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        pre_errors.append(float(np.max(np.abs(actual-expected))))
model=BinaryClassifier()
old=TorchClassifier('models/fall_image2d_multiview/inference.yaml','models/fall_image2d_multiview/best/weights-2-17278.pt','cpu')
video_results=[]
for d in sorted(Path('outputs/mediapipe_fall_image2d').iterdir()):
    if not (d/'poses.npz').exists(): continue
    records=v.load_records(d/'poses.npz')
    meta=json.loads((d/'extraction.json').read_text())
    for r in records:r['ntu25']=map_landmarks(r['image33'])*[meta['width'],meta['height'],0]
    scores=classify_records(records,model,meta['fps'])
    original=list(csv.DictReader((d/'windows.csv').open()))
    assert len(scores)==len(original)
    differences=[]
    for a,b in zip(scores,original):
        assert a['track_id']==int(b['track_id']) and a['end_frame']==int(b['end_frame'])
        assert a['predicted_label']==int(b['predicted_label'])
        differences.append(abs(a['p_fall']-float(b['p_fall'])))
    print(d.name, max(differences,default=0), flush=True)
    assert max(differences,default=0)<1e-3
    video_results.append(dict(video=d.name,windows=len(scores),fall_windows=sum(s['predicted_label'] for s in scores),max_probability_error=max(differences,default=0)))
    if d.name=='20240912_101331':
        firstid=records[0]['track_id']
        poses=np.stack([r['ntu25'][...,:2] for r in records if r['track_id']==firstid])
        np.save(ROOT/'examples/pose_xy.npy',poses)
        np.save(ROOT/'examples/model_input.npy',image_model_input(poses)[None])
poses=np.load(ROOT/'examples/pose_xy.npy')
expected_score=old.score(poses)
actual_score=model.score(poses)
assert abs(actual_score-expected_score)<1e-5
(ROOT/'examples/expected.json').write_text(json.dumps(dict(p_fall=actual_score,pytorch_cpu_p_fall=expected_score),indent=2)+'\n')
report=dict(cpu_example_probability_error=abs(actual_score-expected_score), cached_reference_device='PyTorch CUDA, previous video inference', cached_probability_atol=1e-3, preprocessing_cases=len(pre_errors),max_preprocess_abs_error=max(pre_errors),videos=video_results,passed=True)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
