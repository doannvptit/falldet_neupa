import unittest
from pathlib import Path

import numpy as np

from video_fall.adapter import PoseTracker, map_landmarks, map_visibility, window_ends
from video_fall.classifier import classify_records, model_input


class AdapterTests(unittest.TestCase):
    def test_anatomical_sides_and_virtual_spine(self):
        p = np.zeros((33, 3), np.float32)
        p[11], p[12] = [-1, -2, .3], [1, -2, .3]
        p[23], p[24] = [-.5, 0, .1], [.5, 0, .1]
        p[13], p[14] = [-2, -1, .4], [2, -1, .4]
        ntu = map_landmarks(p, camera_axes=True)
        np.testing.assert_allclose(ntu[0], [0, 0, .1])
        np.testing.assert_allclose(ntu[1], [0, 1, .2])
        np.testing.assert_allclose(ntu[20], [0, 2, .3])
        np.testing.assert_allclose(ntu[5], [2, 1, .4])  # Left elbow retained, X flipped
        np.testing.assert_allclose(ntu[9], [-2, 1, .4])

    def test_visibility_is_conservative_and_invalid_points_rejected(self):
        v = np.ones(33, np.float32)
        v[23] = .1
        self.assertAlmostEqual(float(map_visibility(v)[0]), .1, places=6)
        with self.assertRaises(ValueError):
            map_landmarks(np.full((33, 3), np.nan))

    def test_tracker_keeps_ids_when_detection_order_changes_and_expires(self):
        a, b = np.zeros((33, 3)), np.zeros((33, 3))
        a[:, 0], b[:, 0] = .1, .8
        tracker = PoseTracker(max_gap=2)
        self.assertEqual(tracker.update(0, [a, b]), [1, 2])
        self.assertEqual(tracker.update(1, [b, a]), [2, 1])
        self.assertEqual(tracker.update(5, [a]), [3])

    def test_windows_include_short_clip_tail_without_future_frames(self):
        self.assertEqual(window_ends(0, 10), [])
        self.assertEqual(window_ends(0, 17), [15, 17])
        self.assertEqual(window_ends(0, 55), [15, 23, 31, 39, 47, 55])

    def test_gap_segments_and_minimum_observations(self):
        class FakeClassifier:
            def score(self, poses):
                self.last = poses
                return .8
        fake = FakeClassifier()
        p = np.ones((25, 3), np.float32)
        records = [dict(frame=f, track_id=1, ntu25=p) for f in list(range(18))+list(range(40, 58))]
        results = classify_records(records, fake, fps=30)
        self.assertEqual([r['end_frame'] for r in results], [15, 17, 55, 57])
        self.assertEqual([r['start_frame'] for r in results], [0, 0, 40, 40])
        self.assertTrue(all(r['predicted_label'] == 1 for r in results))
        self.assertEqual(classify_records(records[:5], fake, fps=30), [])
        self.assertEqual(classify_records([], fake, fps=30), [])

    def test_missing_frames_preserve_source_time(self):
        class FakeClassifier:
            def score(self, poses):
                self.last = poses
                return .2
        fake = FakeClassifier()
        p = np.ones((25, 3), np.float32)
        records = [dict(frame=f, track_id=1, ntu25=p) for f in range(20) if f not in (5, 6)]
        results = classify_records(records, fake, fps=30)
        self.assertEqual(results[-1]['end_frame'], 19)
        self.assertEqual(results[-1]['observed_frames'], 18)
        self.assertTrue(np.isnan(fake.last[5:7]).all())
        self.assertTrue(all(r['predicted_label'] == 0 for r in results))


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_matches_actual_training_sample_input(self):
        from data_gen.prepare_ntu120_fall_v2 import read_skeleton_sequence
        source = Path('data/skeleton_extracted/nturgb+d_skeletons/S001C001P003R001A043.skeleton')
        expected = Path('outputs/fall_sample/model_input.npy')
        if not source.exists() or not expected.exists():
            self.skipTest('Local NTU fixture not available')
        frames = read_skeleton_sequence(source)
        body_id = next(iter(frames[0]))
        poses = np.stack([frame[body_id] for frame in frames])
        actual = model_input(poses)
        self.assertEqual(actual.shape, (3, 64, 25, 2))
        np.testing.assert_array_equal(actual, np.load(expected)[0])

    def test_valid_origin_joint_survives_centering(self):
        # This is a legitimate hip-relative origin, not a missing joint.
        rng = np.random.default_rng(4)
        poses = rng.normal(size=(20, 25, 3)).astype(np.float32)
        poses[:, 0] = 0
        actual = model_input(poses)
        shifted = model_input(poses + np.array([0, 0, 4], np.float32))
        np.testing.assert_array_equal(actual, shifted)
        self.assertTrue(np.isfinite(actual).all())
        with self.assertRaises(ValueError):
            model_input(np.full((20, 25, 3), np.nan))


if __name__ == '__main__':
    unittest.main()
