"""Standalone ONNX Runtime binary classifier; no training code or PyTorch."""
from pathlib import Path
import numpy as np
from .preprocess import image_model_input
from .runtime import create_session

DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'models/fall_image2d.onnx'


class BinaryClassifier:
    def __init__(self, model=DEFAULT_MODEL, device='cpu', threads=4,
                 temporal_method='legacy', interpolation_gap=2):
        if temporal_method not in ('legacy','linear','nearest') or interpolation_gap<0:
            raise ValueError('Invalid temporal interpolation options')
        self.temporal_method,self.interpolation_gap=temporal_method,interpolation_gap
        self.session = create_session(model,device,threads)
        self.device = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, tensor):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.shape != (1, 3, 64, 25, 2):
            raise ValueError('Expected preprocessed (1,3,64,25,2) tensor')
        if not np.isfinite(tensor).all() or np.any(tensor[:, 2] != 0):
            raise ValueError('Input must be finite with Z exactly zero')
        return self.session.run(['logits'], {self.input_name: np.ascontiguousarray(tensor)})[0]

    def predict(self, poses=None, *, tensor=None, threshold=.5):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be in [0,1]')
        if (poses is None) == (tensor is None):
            raise ValueError('Provide exactly one of poses or tensor')
        if tensor is None:
            if self.temporal_method=='legacy':
                tensor = image_model_input(poses)[None]
            else:
                from .temporal import temporal_model_input
                tensor=temporal_model_input(poses,self.temporal_method,self.interpolation_gap)[None]
        logits = self.logits(tensor)
        probs = np.exp(logits-logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        score = float(probs[0, 1])
        label = int(score >= threshold)
        return dict(p_fall=score, predicted_label=label,
                    label='FALL' if label else 'NON-FALL', logits=logits[0].tolist())

    def score(self, poses):
        return self.predict(poses)['p_fall']
