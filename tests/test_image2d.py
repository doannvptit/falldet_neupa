import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from feeders.feeder_fall import Feeder
from pose_2d import virtual_camera
from video_fall.classifier import image_model_input


class Image2DTests(unittest.TestCase):
    def test_full_eval_finds_weights_in_best_directory(self):
        from eval_fall import find_best_weights
        with tempfile.TemporaryDirectory() as d:
            best = Path(d)/'best'
            best.mkdir()
            expected = best/'weights-2-17278.pt'
            expected.touch()
            (best/'checkpoint-2-17278.pt').touch()
            (best/'acc.txt').touch()
            self.assertEqual(Path(find_best_weights(d)), expected)

    def test_training_data_keeps_camera_vertical_and_depth_before_projection(self):
        from data_gen.prepare_ntu120_fall_v2 import build_sample
        pose = np.ones((25, 3), np.float32)
        pose[:, 2] = 4
        pose[0], pose[1] = [1, 1, 4], [1, 1.5, 4]
        pose[3] = [1, 2, 4.2]
        out = build_sample([{1: pose}], 150, 'camera_xyz')
        np.testing.assert_array_equal(out[0, 0], (pose-pose[1]).astype(np.float16))
        np.testing.assert_array_equal(out[0, 0, 1]-out[0, 0, 0], [0, .5, 0])
        self.assertNotEqual(float(out[0, 0, 3, 2]), 0)

    def test_second_person_only_crop_can_be_normalized(self):
        from pose_2d import normalize_image_xy
        data = np.zeros((3, 64, 25, 2), np.float32)
        data[0, :, :, 1] = np.arange(25)
        data[1, :, :, 1] = np.arange(25)[::-1]
        out = normalize_image_xy(data)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.any(out[:, :, :, 1]))
        self.assertFalse(np.any(out[:, :, :, 0]))
        self.assertFalse(np.any(out[2]))

    def test_rgb_extraction_does_not_read_landmark_z_or_world_pose(self):
        from unittest.mock import MagicMock
        from infer_fall_video import extract
        class Point:
            x, y, visibility = .4, .6, 1.
            @property
            def z(self):
                raise AssertionError('Image-only extraction read depth')
        class Result:
            pose_landmarks = [[Point() for _ in range(33)]]
            @property
            def pose_world_landmarks(self):
                raise AssertionError('Image-only extraction read world pose')
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {1: 30., 2: 320, 3: 240}[prop]
        cap.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        cv2 = MagicMock(CAP_PROP_FPS=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3)
        cv2.VideoCapture.return_value = cap
        cv2.cvtColor.return_value = cap.read.return_value[1]
        mp = MagicMock()
        pose = mp.tasks.vision.PoseLandmarker.create_from_options.return_value.__enter__.return_value
        pose.detect_for_video.return_value = Result()
        args = SimpleNamespace(pose_model=Path('/tmp/fake.task'), pose_confidence=.5,
                               num_poses=2, max_gap=8, max_frames=1,
                               input_coordinate_mode='image2d_multiview')
        records, meta = extract(Path('/tmp/fake.mp4'), args, cv2, mp)
        self.assertEqual(meta['frames'], 1)
        self.assertEqual(len(records), 1)
        self.assertFalse(np.any(records[0]['world33']))
        self.assertFalse(np.any(records[0]['image33'][:, 2]))
        self.assertFalse(np.any(records[0]['ntu25'][:, 2]))

    def test_camera_foreshortening_fixed_per_clip_and_zero_padding(self):
        data = np.zeros((20, 2, 25, 3), np.float32)
        data[:, 0, 0] = [0, 1, 0]
        top = virtual_camera(data, 'overhead')
        side = virtual_camera(data, 'side_left')
        self.assertGreater(abs(side[0, 0, 0, 1]), 3*abs(top[0, 0, 0, 1]))
        np.testing.assert_array_equal(top[0], top[-1])
        self.assertFalse(np.any(top[:, 1]))
        self.assertFalse(np.any(top[..., 2]))

    def test_inference_ignores_depth_and_is_translation_scale_invariant(self):
        rng = np.random.default_rng(123)
        poses = rng.uniform(100, 400, (30, 25, 3)).astype(np.float32)
        first = image_model_input(poses)
        poses[..., 2] = np.nan
        np.testing.assert_array_equal(first, image_model_input(poses))
        poses[..., 2] = np.inf
        np.testing.assert_array_equal(first, image_model_input(poses))
        moved = poses.copy()
        moved[..., :2] = moved[..., :2]*2 + [100, 200]
        np.testing.assert_allclose(first, image_model_input(moved), atol=.002)
        self.assertFalse(np.any(first[2]))

    def test_three_eval_views_and_image_inference_match(self):
        rng = np.random.default_rng(12)
        sample = rng.normal(0, .3, (150, 2, 25, 3)).astype(np.float16)
        sample[30:] = 0
        sample[:, 1] = 0
        sample[:, :, 1] = 0
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'stats.json').write_text('{"coordinate_mode":"camera_xyz"}')
            for split in ('train', 'test'):
                np.save(Path(d)/f'x_{split}.npy', sample[None])
                np.save(Path(d)/f'y_{split}.npy', np.array([1]))
            ds = Feeder(d, split='test', window_size=64, p_interval=[.95],
                        multi_input=False, zero_z=True, image2d_multiview=True)
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds.label.tolist(), [1, 1, 1])
            for index, view in enumerate(ds.eval_views):
                x, _, _ = ds[index]
                pixels = virtual_camera(sample.astype(np.float32), view)[:30, 0]
                expected = image_model_input(pixels)
                np.testing.assert_allclose(x.numpy(), expected, atol=.002)
            train = Feeder(d, split='train', window_size=64, p_interval=[.8, 1],
                           random_rot=True, multi_input=False, zero_z=True, image2d_multiview=True)
            for _ in range(10):
                x, _, _ = train[0]
                self.assertTrue(np.isfinite(x.numpy()).all())
                self.assertFalse(np.any(x[2].numpy()))


if __name__ == '__main__':
    unittest.main()
