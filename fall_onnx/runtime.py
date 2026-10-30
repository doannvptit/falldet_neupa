"""Shared explicit ONNX providers and bounded CPU threads."""
import onnxruntime as ort


def create_session(path, device='cpu', threads=4):
    if threads < 1:
        raise ValueError('threads must be positive')
    if device not in ('cpu','cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda':
        if hasattr(ort,'preload_dlls'):
            ort.preload_dlls()  # CUDA/cuDNN from system or NVIDIA pip packages.
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA provider unavailable; install requirements-gpu.txt or use --device cpu')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    providers = ['CPUExecutionProvider'] if device == 'cpu' else [
        ('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider']
    session = ort.InferenceSession(str(path),sess_options=options,providers=providers)
    if device == 'cuda' and session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA initialization failed; check CUDA/cuDNN libraries (no silent CPU fallback)')
    return session
