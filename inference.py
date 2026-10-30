"""Classify one NTU25 pixel-XY sequence or a preprocessed ONNX tensor."""
import argparse
import json
from pathlib import Path
import numpy as np
from fall_onnx.classifier import BinaryClassifier, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='.npy (T,25,2/3), pixels; Z ignored')
    parser.add_argument('--preprocessed', action='store_true', help='Input already has shape (1,3,64,25,2)')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--temporal-method',choices=['legacy','linear','nearest'],default='legacy')
    parser.add_argument('--interpolation-gap',type=int,default=2)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = BinaryClassifier(args.model,args.device,args.threads,args.temporal_method,args.interpolation_gap)
    data = np.load(args.input, allow_pickle=False)
    result = model.predict(**({'tensor': data} if args.preprocessed else {'poses': data}), threshold=args.threshold)
    result.update(provider=model.device, threshold=args.threshold)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
