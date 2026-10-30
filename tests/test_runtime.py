"""Run: python -m unittest discover -s tests -v (no training dependencies)."""
import json
from pathlib import Path
import unittest
import numpy as np
from fall_onnx.classifier import BinaryClassifier
from fall_onnx.preprocess import image_model_input
from fall_onnx.windows import classify_records
from fall_onnx.adapter import map_landmarks

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BinaryClassifier()
        cls.poses = np.load(ROOT/'examples/pose_xy.npy')
        cls.tensor = np.load(ROOT/'examples/model_input.npy')

    def test_reference_preprocessing(self):
        np.testing.assert_allclose(image_model_input(self.poses)[None], self.tensor, atol=1e-6)

    def test_reference_prediction(self):
        expected = json.loads((ROOT/'examples/expected.json').read_text())['p_fall']
        raw = self.model.predict(self.poses)
        tensor = self.model.predict(tensor=self.tensor)
        self.assertAlmostEqual(raw['p_fall'], expected, places=5)
        self.assertEqual(raw['predicted_label'], tensor['predicted_label'])
        self.assertEqual(raw['predicted_label'], 1)

    def test_depth_is_ignored(self):
        poses = np.concatenate([self.poses,np.full((*self.poses.shape[:2],1),np.inf)],axis=-1)
        np.testing.assert_array_equal(image_model_input(poses),image_model_input(self.poses))
        landmarks = np.ones((33,3),np.float32)
        landmarks[:,2] = np.nan
        self.assertTrue(np.all(map_landmarks(landmarks)[:,2] == 0))

    def test_invalid_pose_rejected(self):
        for poses in [np.zeros((16,25,2)),np.full((16,25,2),np.nan),np.zeros((16,33,2))]:
            with self.assertRaises(ValueError):
                image_model_input(poses)

    def test_tensor_contract(self):
        with self.assertRaises(ValueError):
            self.model.logits(self.tensor[0])
        x = self.tensor.copy()
        x[:,2] = 1
        with self.assertRaises(ValueError):
            self.model.logits(x)

    def test_gaps_and_causal_endpoints(self):
        class Dummy:
            def score(self, poses):
                return .75
        # Do not join two segments with a 25-frame occlusion.
        records = [dict(frame=i,track_id=1,ntu25=np.ones((25,3)))
                   for i in [*range(20),*range(45,65)]]
        scores = classify_records(records,Dummy(),30)
        self.assertEqual([s['end_frame'] for s in scores],[15,19,60,64])
        self.assertTrue(all(s['observed_frames']==s['end_frame']-s['start_frame']+1 for s in scores))
        self.assertEqual(classify_records(records[:10],Dummy(),30),[])


if __name__ == '__main__':
    unittest.main()
