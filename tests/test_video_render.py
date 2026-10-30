"""Render checks use tiny synthetic video, not claimed as RGB pose validation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from infer_fall_video import load_records, render, save_records


class CacheTests(unittest.TestCase):
    def test_empty_and_nonempty_cache_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'poses.npz'
            save_records(path, [])
            self.assertEqual(load_records(path), [])
            record = dict(frame=4, track_id=2, image33=np.zeros((33, 3), np.float32),
                          world33=np.ones((33, 3), np.float32), visibility33=np.ones(33, np.float32),
                          ntu25=np.ones((25, 3), np.float32), visibility25=np.ones(25, np.float32))
            save_records(path, [record])
            actual = load_records(path)[0]
            self.assertEqual(actual['frame'], 4)
            self.assertEqual(actual['track_id'], 2)
            np.testing.assert_array_equal(actual['ntu25'], record['ntu25'])


@unittest.skipUnless(importlib.util.find_spec('cv2') and shutil.which('ffmpeg'), 'Requires OpenCV and ffmpeg')
class RenderingTests(unittest.TestCase):
    def test_causal_overlay_no_pose_and_warmup_and_h264_frame_count(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source, output = tmp/'source.mp4', tmp/'annotated.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=c=black:s=320x240:r=10', '-frames:v', '20',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            records = [dict(frame=f, track_id=1, image33=np.full((33, 3), .5, np.float32),
                            visibility25=np.ones(25)) for f in range(1, 20)]
            scores = [dict(track_id=1, end_frame=15, p_fall=.9, predicted_label=1)]
            timeline = render(source, output, records, scores,
                              dict(width=320, height=240, fps=10, frames=20),
                              SimpleNamespace(stride=8, max_gap=8, pose_confidence=.5), cv2)
            self.assertEqual(timeline[0]['people'], [])
            self.assertEqual(timeline[14]['people'][0]['state'], 'WARMUP')
            self.assertIsNone(timeline[14]['people'][0]['p_fall'])
            self.assertEqual(timeline[15]['people'][0]['state'], 'FALL')
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=codec_name,nb_frames', '-of', 'json', str(output)]))
            self.assertEqual(info['streams'][0]['codec_name'], 'h264')
            self.assertEqual(int(info['streams'][0]['nb_frames']), 20)


if __name__ == '__main__':
    unittest.main()
