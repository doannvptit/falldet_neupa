"""One-time export from the training project; runtime bundle needs no PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from video_fall.classifier import BinaryClassifier


class FusedHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, skeleton):
        return self.model(skeleton)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT.parent/'poc_inference_onnx/models/fall_image2d.onnx')
    args = ap.parse_args()
    config = ROOT/'models/fall_image2d_multiview/inference.yaml'
    weights = ROOT/'models/fall_image2d_multiview/best/weights-2-17278.pt'
    model = FusedHead(BinaryClassifier(config, weights, 'cpu').model).eval()
    example = np.load(ROOT/'outputs/multiview_2d_training/image2d_model_input.npy')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(example), str(args.output),
                          input_names=['skeleton'], output_names=['logits'],
                          opset_version=17, dynamo=False, external_data=False)
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(2026)
    cases = [example, np.zeros_like(example)]
    for _ in range(5):
        x = rng.normal(size=example.shape).astype(np.float32)
        x[:, 2] = 0
        cases.append(x)
    errors = []
    with torch.inference_mode():
        for x in cases:
            expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(['logits'], {'skeleton': x})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            errors.append(float(np.max(np.abs(actual-expected))))
    report = dict(model='EGGCN image2d multiview, binary fused logits',
        input=dict(name='skeleton',shape=[1,3,64,25,2],dtype='float32',z='exactly zero'),
        output=dict(name='logits',shape=[1,2],labels={'0':'NON-FALL','1':'FALL'},postprocess='softmax, class 1'),
        opset=17, checkpoint=str(weights), checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()), torch_version=torch.__version__,
        onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
        verification=dict(cases=len(cases),max_abs_logit_error=max(errors),rtol=1e-4,atol=1e-4,passed=True))
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
