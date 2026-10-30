"""Download pinned official OpenMMLab ONNX assets. Runtime stays offline."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
 'yolox_s': ('yolox_s_8xb8-300e_humanart-3ef259a7.zip', [1,3,640,640]),
 'rtmpose_m_halpe26': ('rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', [1,3,256,192]),
}
BASE='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',type=Path,default=ROOT/'models')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    args.model_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.model_dir/'pose_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name,(archive,shape) in ASSETS.items():
        destination=args.model_dir/(name+'.onnx')
        if destination.exists() and not args.force:
            sha=hashlib.sha256(destination.read_bytes()).hexdigest()
            if manifest.get(name,{}).get('sha256')!=sha:
                raise RuntimeError(f'Existing asset not verified: {destination}; use --force')
            print(f'Verified {destination}',flush=True)
            continue
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.zip'
            errors=[]
            # The maintainer mirrors the same official archive if OpenMMLab is unavailable.
            urls=[BASE+archive,'https://huggingface.co/Tau-J/RTMPose/resolve/main/'+archive]
            for url in urls:
                try:
                    print(f'Downloading {url}',flush=True)
                    with urllib.request.urlopen(url,timeout=120) as response, path.open('wb') as out:
                        while chunk:=response.read(1024*1024): out.write(chunk)
                    with zipfile.ZipFile(path) as z:
                        members=[n for n in z.namelist() if n.endswith('.onnx')]
                        if len(members)!=1: raise ValueError(f'Expected one ONNX file: {members}')
                        content=z.read(members[0])
                    break
                except Exception as exc:
                    errors.append(str(exc))
            else:
                raise RuntimeError('Model download failed: '+'; '.join(errors))
            tmp=destination.with_suffix('.onnx.tmp')
            tmp.write_bytes(content)
            tmp.replace(destination)
            manifest[name]=dict(source=url,official_source=BASE+archive,
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sha256=hashlib.sha256(content).hexdigest(),input_shape=shape,
                file=destination.name,bytes=len(content))
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            print(f'Saved {destination} ({len(content)/1e6:.1f} MB)',flush=True)

if __name__=='__main__': main()
