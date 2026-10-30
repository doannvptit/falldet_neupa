import unittest
import numpy as np
from fall_onnx.temporal import resample_xy,temporal_model_input
from fall_onnx.windows import classify_records


class TemporalTests(unittest.TestCase):
    def test_linear_midpoints_and_nearest_control(self):
        p=np.ones((2,25,3),np.float32)*10
        p[1,:,:2]=30;p[:,:,2]=np.inf
        a=resample_xy(p,5,'linear')
        np.testing.assert_array_equal(a[:,0,0],[10,15,20,25,30])
        np.testing.assert_array_equal(resample_xy(p,5,'nearest')[:,0,0],[10,10,10,30,30])
        self.assertTrue(np.all(a[...,2]==0))

    def test_short_gap_interpolates_but_long_gap_stays_missing(self):
        p=np.zeros((6,25,2),np.float32)
        p[0]=10;p[-1]=30
        a=resample_xy(p,11,'linear',max_gap=2)
        np.testing.assert_array_equal(a[1:-1],0)
        b=resample_xy(p[:3],5,'linear',max_gap=2)
        np.testing.assert_array_equal(b[1:],0)  # no extrapolation after one observation
        p[2]=20
        c=resample_xy(p[:3],5,'linear',max_gap=1)
        np.testing.assert_array_equal(c[:,0,0],[10,12.5,15,17.5,20])

    def test_missing_one_joint_does_not_erase_other_joints(self):
        p=np.ones((2,25,2),np.float32)*10
        p[1]=30;p[0,4]=np.nan
        a=resample_xy(p,3)
        np.testing.assert_array_equal(a[:2,4],0)
        np.testing.assert_array_equal(a[:,5,0],[10,20,30])

    def test_prepared_input_contract_and_ignored_depth(self):
        rng=np.random.default_rng(8)
        p=rng.uniform(1,1000,(16,25,3)).astype(np.float32)
        a=temporal_model_input(p)
        p[...,2]=np.nan
        b=temporal_model_input(p)
        self.assertEqual(a.shape,(3,64,25,2))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(a[2]==0))

    def test_full_window_waits_and_never_reads_future(self):
        class Spy:
            def __init__(self): self.last=[]
            def score(self,poses):
                self.last.append(float(poses[-1,0,0]))
                return .75
        records=[dict(track_id=1,frame=i,ntu25=np.ones((25,3),np.float32)*(i+1)) for i in range(80)]
        spy=Spy()
        scores=classify_records(records,spy,8,window=64,stride=2,min_frames=8,
                                max_gap=2,require_full_window=True)
        self.assertEqual(scores[0]['end_frame'],63)
        self.assertEqual(scores[0]['end_sec'],7.875)
        self.assertEqual(spy.last,[s['end_frame']+1 for s in scores])
        self.assertEqual(classify_records(records[:16],spy,8,window=64,min_frames=8,
                                         require_full_window=True),[])

    def test_synthetic_frames_never_inflate_observed_coverage(self):
        class Dummy:
            def score(self,poses):return .8
        records=[dict(track_id=1,frame=i,ntu25=np.ones((25,3),np.float32)) for i in range(16) if i not in (4,5)]
        scores=classify_records(records,Dummy(),8,window=16,stride=2,min_frames=8,max_gap=2)
        self.assertEqual(scores[-1]['observed_frames'],14)
        self.assertEqual(scores[-1]['coverage'],14/16)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):resample_xy(np.ones((2,17,2)))
        with self.assertRaises(ValueError):resample_xy(np.ones((2,25,2)),method='cubic')
        with self.assertRaises(ValueError):temporal_model_input(np.zeros((8,25,2)))


if __name__=='__main__': unittest.main()
