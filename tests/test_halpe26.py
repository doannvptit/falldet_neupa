import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from fall_onnx.halpe26 import map_halpe26, DIRECT
from fall_onnx.pose import PoseEstimator, _Detector
from fall_onnx.adapter import PoseTracker
from infer_video import validate_cache


class HalpeTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.arange(52,dtype=np.float32).reshape(26,2)+100
        self.scores = np.ones(26,np.float32)

    def test_direct_joints_and_exact_zero_z(self):
        out, scores = map_halpe26(self.xy,self.scores)
        for target,source in DIRECT.items():
            np.testing.assert_array_equal(out[target,:2],self.xy[source])
            self.assertEqual(scores[target],1)
        np.testing.assert_array_equal(out[[0,2,3],:2],self.xy[[19,18,17]])
        self.assertTrue(np.all(out[:,2]==0))

    def test_spine_feet_and_hand_geometry(self):
        out,scores = map_halpe26(self.xy,self.scores)
        np.testing.assert_allclose(out[20,:2],self.xy[[5,6]].mean(0))
        np.testing.assert_allclose(out[1,:2],(self.xy[19]+self.xy[[5,6]].mean(0))/2)
        np.testing.assert_allclose(out[15,:2],self.xy[[20,22]].mean(0))
        np.testing.assert_allclose(out[19,:2],self.xy[[21,23]].mean(0))
        np.testing.assert_allclose(out[21,:2],self.xy[9]+.3*(self.xy[9]-self.xy[7]))
        self.assertEqual(scores[21],.5)

    def test_translation_equivariance(self):
        a,_=map_halpe26(self.xy,self.scores)
        b,_=map_halpe26(self.xy+[20,35],self.scores)
        np.testing.assert_allclose(b[:,:2]-a[:,:2],np.tile([20,35],(25,1)),atol=2e-5)

    def test_missing_wrist_masks_dependent_hand_joints(self):
        self.scores[9]=.1
        out,scores=map_halpe26(self.xy,self.scores)
        np.testing.assert_array_equal(out[[6,7,21,22]],0)
        np.testing.assert_array_equal(scores[[6,7,21,22]],0)
        self.assertEqual(scores[10],1)

    def test_missing_central_hip_falls_back_to_hips(self):
        self.xy[19]=np.nan
        out,scores=map_halpe26(self.xy,self.scores)
        np.testing.assert_allclose(out[0,:2],self.xy[[11,12]].mean(0))
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(scores[1],0)

    def test_foot_uses_visible_toe_only(self):
        self.scores[[20,21,23]]=0
        out,scores=map_halpe26(self.xy,self.scores)
        np.testing.assert_array_equal(out[15,:2],self.xy[22])
        np.testing.assert_array_equal(out[19],0)

    def test_invalid_layout_rejected(self):
        with self.assertRaises(ValueError): map_halpe26(self.xy[:17],self.scores[:17])

    def test_no_detector_boxes_never_runs_pose(self):
        estimator=object.__new__(PoseEstimator)
        estimator.detector=Mock(return_value=np.empty((0,4)))
        estimator.pose=Mock(side_effect=AssertionError('No detections must not run pose'))
        xy,scores,boxes=estimator(np.zeros((64,64,3),np.uint8))
        self.assertEqual(xy.shape,(0,26,2))
        estimator.pose.assert_not_called()

    def test_embedded_nms_respects_threshold_and_sort(self):
        detector=object.__new__(_Detector)
        detector.score_thr=.5
        boxes=detector.postprocess(np.array([[[1,2,3,4,.6],[5,6,7,8,.4],[9,10,11,12,.9]]]),2)
        np.testing.assert_array_equal(boxes,[[4.5,5,5.5,6],[.5,1,1.5,2]])

    def test_tracker_survives_detection_order_change(self):
        tracker=PoseTracker()
        self.assertEqual(tracker.update_centers(0,[[.1,.2],[.8,.2]]),[1,2])
        self.assertEqual(tracker.update_centers(1,[[.79,.2],[.11,.2]]),[2,1])

    def test_cache_rejects_legacy_and_adapter_change(self):
        with tempfile.TemporaryDirectory() as temp:
            video=Path(temp)/'video.mp4'
            video.write_bytes(b'test')
            signature={'pipeline':'yolox_rtmpose_halpe26','adapter':'v1','pose_sha256':'abc'}
            meta=dict(source=str(video.resolve()),source_size=4,source_mtime_ns=video.stat().st_mtime_ns,
                      extraction_signature=signature)
            validate_cache(meta,video,signature)
            old=copy.deepcopy(meta)
            old.pop('extraction_signature')
            with self.assertRaises(ValueError): validate_cache(old,video,signature)
            with self.assertRaises(ValueError): validate_cache(meta,video,{**signature,'adapter':'v2'})


if __name__=='__main__': unittest.main()
