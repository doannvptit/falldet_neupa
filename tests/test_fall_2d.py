import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.prepare_ntu120_fall_v2 import build_sample
from feeders.feeder_fall import Feeder
from video_fall.classifier import model_input


class PlanarInputTests(unittest.TestCase):
    def test_pca_feeder_matches_video_and_preserves_padding(self):
        from pose_projection import load_projection, project_pose
        rng = np.random.default_rng(123)
        poses = rng.uniform(1, 2, (24, 25, 3)).astype(np.float32)
        sample = build_sample([{1: p} for p in poses], 150)
        components = np.array([[0, 0, 1], [1, 0, 0]], np.float32)
        with tempfile.TemporaryDirectory() as d:
            projection_path = Path(d)/'projection.npz'
            np.savez(projection_path, mean=np.array([.1, .2, .3]), components=components)
            projection = load_projection(projection_path)
            projected = project_pose(sample, projection)
            self.assertTrue(np.all(projected[..., 2] == 0))
            self.assertTrue(np.all(projected[24:] == 0))
            self.assertTrue(np.all(projected[:, 1] == 0))
            np.save(Path(d)/'x_test.npy', sample[None])
            np.save(Path(d)/'y_test.npy', np.array([1]))
            ds = Feeder(d, split='test', p_interval=[.95], window_size=64,
                        multi_input=False, zero_z=True, projection_path=projection_path)
            actual, _, _ = ds[0]
            expected = model_input(poses, projection=projection)
            np.testing.assert_array_equal(actual.numpy(), expected)
            ds.random_rot = True
            augmented, _, _ = ds[0]
            self.assertFalse(np.any(augmented[2].numpy()))

    def test_xy_height_retained_depth_removed_and_inference_matches_feeder(self):
        rng = np.random.default_rng(42)
        poses = rng.uniform(1, 2, (24, 25, 3)).astype(np.float32)
        frames = [{1: p} for p in poses]
        sample = build_sample(frames, 150, 'xy_zero_z')
        self.assertTrue(np.all(sample[..., 2] == 0))
        expected = poses[..., :2] - poses[:, 1:2, :2]
        np.testing.assert_array_equal(sample[:24, 0, :, :2], expected.astype(np.float16))
        changed = poses.copy()
        changed[..., 2] *= 100
        np.testing.assert_array_equal(sample, build_sample([{1: p} for p in changed], 150, 'xy_zero_z'))
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d)/'x_test.npy', sample[None])
            np.save(Path(d)/'y_test.npy', np.array([1]))
            feeder = Feeder(d, split='test', p_interval=[.95], window_size=64,
                            multi_input=False, zero_z=True)
            actual, _, _ = feeder[0]
            predicted_input = model_input(poses, coordinate_mode='xy_zero_z')
            np.testing.assert_array_equal(actual.numpy(), predicted_input)
            augmented = Feeder(d, split='test', p_interval=[.95], window_size=64,
                               multi_input=False, zero_z=True, random_rot=True)
            rotated, _, _ = augmented[0]
            self.assertEqual(int(np.count_nonzero(rotated[2].numpy())), 0)
            self.assertGreater(int(np.count_nonzero(rotated[1].numpy())), 0)


if __name__ == '__main__':
    unittest.main()
