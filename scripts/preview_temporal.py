"""Offline side-by-side observed/linear pose preview, no image synthesis."""
import argparse,csv,json
from pathlib import Path
import subprocess,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from infer_video import load_records
from fall_onnx.temporal import resample_xy,temporal_model_input
EDGES=[(a-1,b-1) for a,b in ((1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(23,8),(24,25),(25,12))]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,default=ROOT/'outputs/eval_8fps')
    ap.add_argument('--video',default='20240912_101331')
    ap.add_argument('--policy',default='linear_2s')
    args=ap.parse_args();root=args.experiment
    rows=list(csv.DictReader((root/args.policy/args.video/'windows.csv').open()))
    if not rows:raise ValueError('No evaluated windows')
    best=max(rows,key=lambda r:float(r['p_fall']))
    start,end,tid=[int(best[k]) for k in ('start_frame','end_frame','track_id')]
    records=load_records(root/'poses_8fps'/args.video/'poses.npz')
    by_frame={r['frame']:r for r in records if r['track_id']==tid}
    observed=np.stack([by_frame[i]['ntu25'] if i in by_frame else np.full((25,3),np.nan,np.float32) for i in range(start,end+1)])
    interpolated=resample_xy(observed,64,'linear',2)
    out=root/'previews'/f'{args.video}_{args.policy}';out.mkdir(parents=True,exist_ok=True)
    np.save(out/'observed_xy.npy',observed)
    np.save(out/'interpolated_64_xy.npy',interpolated)
    np.save(out/'model_input.npy',temporal_model_input(observed,'linear',2)[None])
    cap=cv2.VideoCapture(str(root/'videos_8fps'/(args.video+'.mp4')))
    backgrounds=[]
    for i in range(end+1):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Cannot read preview source')
        if i>=start:backgrounds.append(frame)
    cap.release()
    fps=63*8/max(end-start,1)
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x430',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'interpolated_preview.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for n,pos in enumerate(np.linspace(0,end-start,64)):
            index=min(int(np.floor(pos)),len(backgrounds)-1)
            canvas=np.zeros((430,1280,3),np.uint8)
            base=cv2.resize(backgrounds[index],(640,360))
            canvas[70:,:640]=base;canvas[70:,640:]=base
            for offset,points,color in [(0,observed[index],(70,230,70)),(640,interpolated[n],(0,180,255))]:
                xy=points[:,:2]
                valid=np.isfinite(xy).all(-1)&np.any(xy!=0,axis=-1)
                coords=np.rint(np.nan_to_num(xy)*[640/backgrounds[index].shape[1],360/backgrounds[index].shape[0]]+[offset,70]).astype(int)
                for a,b in EDGES:
                    if valid[a] and valid[b]: cv2.line(canvas,tuple(coords[a]),tuple(coords[b]),color,2,cv2.LINE_AA)
                for point,ok in zip(coords,valid):
                    if ok:cv2.circle(canvas,tuple(point),3,color,-1,cv2.LINE_AA)
            for text,y in [('OFFLINE pose interpolation preview; all poses available only at window end',20),
                           (f'Source window {start/8:.3f} - {end/8:.3f}s | pose time {(start+pos)/8:.3f}s',43)]:
                cv2.putText(canvas,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,'Left: observed 8 FPS pose, held',(12,64),cv2.FONT_HERSHEY_SIMPLEX,.48,(70,230,70),1)
            cv2.putText(canvas,'Right: linear XY, 64 poses; RGB held',(652,64),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,180,255),1)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError('Preview encoding failed')
    except BaseException:
        proc.kill();proc.wait();raise
    metadata=dict(track_id=tid,start_frame=start,end_frame=end,source_fps=8,
        observed_timeline_frames=len(observed),model_frames=64,preview_fps=fps,
        decision_available_at_sec=end/8,offline_preview=True,rgb_interpolated=False,
        pose_times_sec=np.linspace(start/8,end/8,64).tolist())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out/'interpolated_preview.mp4')

if __name__=='__main__':main()
