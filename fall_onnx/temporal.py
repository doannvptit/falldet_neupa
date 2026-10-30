"""Time interpolation inside an observed causal window, with bounded gaps."""
import numpy as np
from .preprocess import normalize_image_xy


def resample_xy(poses, count=64, method='linear', max_gap=2):
    """(T,25,2/3) -> (count,25,3), keeping missing joints zero and Z=0.

    Interpolate only between valid observations, never beyond either endpoint
    or over more than max_gap missing source frames. All source frames must
    already be available: intermediate poses are not new measurements.
    """
    poses=np.asarray(poses,np.float32)
    if poses.ndim!=3 or poses.shape[1]!=25 or poses.shape[2] not in (2,3) or not len(poses):
        raise ValueError('Expected nonempty (T,25,2/3)')
    if count<1 or max_gap<0 or method not in ('linear','nearest'):
        raise ValueError('Invalid resampling count/method/max_gap')
    xy=poses[...,:2]
    valid=np.isfinite(xy).all(-1)&np.any(xy!=0,axis=-1)
    positions=np.linspace(0,len(poses)-1,count,dtype=np.float64)
    out=np.zeros((count,25,3),np.float32)
    for joint in range(25):
        observed=np.flatnonzero(valid[:,joint])
        if not len(observed): continue
        for t,pos in enumerate(positions):
            at=np.searchsorted(observed,pos)
            if at<len(observed) and observed[at]==pos:
                out[t,joint,:2]=xy[observed[at],joint]
                continue
            if at==0 or at==len(observed): continue
            left,right=observed[at-1],observed[at]
            if right-left-1>max_gap: continue
            fraction=(pos-left)/(right-left)
            if method=='nearest':
                out[t,joint,:2]=xy[left if fraction<=.5 else right,joint]
            else:
                out[t,joint,:2]=xy[left,joint]*(1-fraction)+xy[right,joint]*fraction
    return out


def temporal_model_input(poses, method='linear', max_gap=2):
    """Pixel XY -> center, float16 roundtrip, crop 95%, resample, normalize.

    Preserve missing-frame positions rather than compacting time or holding
    arbitrarily long occlusions. A new model window is scored only when its
    latest real source frame has arrived; interpolation adds no future frames.
    """
    poses=np.asarray(poses,np.float32)
    if poses.ndim!=3 or poses.shape[1]!=25 or poses.shape[2] not in (2,3):
        raise ValueError('Expected (T,25,2/3)')
    xy=poses[...,:2]
    valid=np.isfinite(xy).all(-1)&np.any(xy!=0,axis=-1)
    frames=np.flatnonzero(valid[:,1])  # observed spine-mid center
    if not len(frames): raise ValueError('No usable spine center')
    first,last=int(frames[0]),int(frames[-1])
    xy=xy[first:last+1][:150].copy()
    valid=valid[first:last+1][:150]
    center_ok=valid[:,1]
    centered=np.zeros((len(xy),25,3),np.float32)
    mask=valid&center_ok[:,None]
    centered[...,:2]=np.where(mask[...,None],xy-xy[:,1:2],0)
    # A centered spine-mid equals zero but is a valid reference point. It
    # remains zero during interpolation; other joint masks stay explicit.
    with np.errstate(over='ignore'):
        centered=centered.astype(np.float16).astype(np.float32)
    if not np.isfinite(centered).all(): raise ValueError('XY overflow')
    bias=int((1-.95)*len(centered)/2)
    cropped=centered[bias:len(centered)-bias]
    result=resample_xy(cropped,64,method,max_gap)
    data=np.zeros((3,64,25,2),np.float32)
    data[:,:,:,0]=result.transpose(2,0,1)
    return normalize_image_xy(data)
