"""Validate completed multiview run, package weights, and write its report."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import yaml


def main():
    root = Path('work_dir/fall/image2d_multiview_2epoch')
    full = json.loads((root/'full_test/fall_test_metrics.json').read_text())
    videos = json.loads(Path('outputs/mediapipe_fall_image2d/summary.json').read_text())
    train_log = (root/'train/log/train.log').read_text()
    rows = re.findall(r'\[fall=1\] TP (\d+) FP (\d+) FN (\d+) TN (\d+) \| Precision ([\d.]+)% \| Recall ([\d.]+)% \| F1 ([\d.]+)%', train_log)
    assert len(rows) == 2 and len(videos) == 4
    weights = Path(full['weights'])
    cfg = yaml.safe_load(Path(full['config']).read_text())
    verification = []
    for video in videos:
        assert video['input_coordinate_mode'] == 'image2d_multiview'
        assert video['projection_path'] is None
        out = Path(video['output'])
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,nb_read_frames,width,height',
            '-of', 'json', str(out)]))['streams'][0]
        assert probe['codec_name'] == 'h264' and int(probe['nb_read_frames']) == video['frames']
        assert len((out.parent/'frames.jsonl').read_text().splitlines()) == video['frames']
        with np.load(out.parent/'poses.npz', allow_pickle=False) as poses:
            assert not np.any(poses['world33']) and not np.any(poses['image33'][..., 2])
            assert not np.any(poses['ntu25'][..., 2])
        scores = list(csv.DictReader((out.parent/'windows.csv').open()))
        assert len(scores) == video['windows']
        verification.append(dict(video=Path(video['video']).name, **probe,
                                 frames=video['frames'], windows=len(scores),
                                 fall_windows=video['fall_windows'],
                                 max_p_fall=max((float(r['p_fall']) for r in scores), default=None),
                                 only_image_xy=True))
    package = Path('models/fall_image2d_multiview')
    (package/'best').mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, package/'best'/weights.name)
    inference = dict(work_dir=str(package), model=cfg['model'], model_args=cfg['model_args'],
                     train_feeder_args=dict(multi_input=False, bone=False, vel=False),
                     test_feeder_args=dict(window_size=64, p_interval=[.95], zero_z=True,
                                           image2d_multiview=True))
    (package/'inference.yaml').write_text(yaml.safe_dump(inference, sort_keys=False))
    metadata = dict(weights=str(weights), sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    input_shape=['B', 3, 64, 25, 2], input_coordinates='image XY pixels; Z ignored/zero',
                    labels={'0': 'non-fall', '1': 'fall'}, threshold=.5,
                    camera_mixture={'overhead': .5, 'side_left': .2, 'side_right': .2, 'oblique': .1},
                    full_synthetic_test=full, video_verification=verification,
                    note='Synthetic camera evaluation; RGB videos have no timing/person ground truth.')
    (package/'metadata.json').write_text(json.dumps(metadata, indent=2))
    (root/'verification.json').write_text(json.dumps(verification, indent=2))
    if Path('/tmp/falldet_multiview_tests.log').exists():
        shutil.copy2('/tmp/falldet_multiview_tests.log', root/'tests.log')
    lines = ['# Kết quả fine-tune image 2D đa góc nhìn', '',
             'Đã chạy đủ 2 epoch trên RTX 3060. Model chỉ nhận X/Y ảnh, Z=0; không cần world pose hay PCA khi inference.', '',
             'Dữ liệu camera XYZ được tạo lại từ 56.578 raw clip, 0 lỗi; giữ trục camera, không xoay theo thân người. '
             'Train: 40.091 clip (671 fall). Mỗi epoch lấy 6.039 mẫu, bỏ batch cuối chưa đủ. '
             'Camera train: 50% trên xuống, 20% trái, 20% phải, 10% xiên khác. '
             'Khởi tạo từ checkpoint binary 3D epoch 2 cũ; LR 0,003, batch 64/forward 16.', '',
             '## Theo dõi mỗi epoch: 2.475 clip × 3 góc = 7.425 phép chiếu', '',
             '| Epoch | TP | FP | FN | TN | Precision | Recall | F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for i, row in enumerate(rows, 1):
        lines.append('| '+str(i)+' | '+' | '.join(row)+' |')
    lines += ['', f'Checkpoint tốt nhất: `{weights}`.', '',
              '## Toàn bộ test: 16.487 clip × 3 góc = 49.461 phép chiếu', '',
              'Cùng clip được chiếu ba lần, không phải 49.461 clip độc lập. Ngưỡng 0,5. '
              'Split test này cũng được dùng để chọn checkpoint qua subset theo dõi; đây không phải bộ holdout độc lập mới.', '',
              '| Góc mô phỏng | Precision | Recall | F1 | TN / FP / FN / TP |',
              '|---|---:|---:|---:|---|']
    for view, m in [('Tổng hợp', full['metrics']), *full['synthetic_view_metrics'].items()]:
        cm = m['cm']
        lines.append(f"| {view} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {cm[0][0]} / {cm[0][1]} / {cm[1][0]} / {cm[1][1]} |")
    lines += ['', '## Video RGB thật: chỉ dùng landmark ảnh X/Y', '',
              '| Video | Frame | Cửa sổ | Cửa sổ FALL | p_fall lớn nhất |',
              '|---|---:|---:|---:|---:|']
    for v in verification:
        lines.append(f"| {v['video']} | {v['frames']} | {v['windows']} | {v['fall_windows']} | {v['max_p_fall']:.6g} |")
    lines += ['', 'Cửa sổ chồng lấn không phải số sự kiện ngã. Chưa có ground truth theo người/thời gian, '
              'nên không suy ra accuracy/precision/recall RGB từ bảng này. Cần video thực có nhãn ở góc trên xuống và hai bên.', '',
              'Đã kiểm tra: 4 video H.264 đủ frame; timeline/windows khớp; cache image Z và world pose đều zero. '
              'Các test gồm đọc landmark không có thuộc tính Z/world, Z NaN/inf không ảnh hưởng input, '
              'giữ đúng trục camera, ba view và crop chỉ còn người thứ hai.', '',
              f'Gói inference: `{package}` (weights, inference.yaml, metadata.json). '
              'Không cần dataset NTU hoặc file PCA để nạp classifier từ gói này. '
              'Video: `outputs/mediapipe_fall_image2d/<tên clip>/annotated.mp4`.', '',
              'Hướng dẫn: `FINETUNE_MULTIVIEW_2D.md`. Ví dụ hình học: `outputs/multiview_2d_training/sample_views.png`.']
    standalone = package/'standalone_validation.json'
    if standalone.exists():
        check = json.loads(standalone.read_text())
        lines += ['', 'Kiểm tra gói inference độc lập: nạp bằng `inference.yaml` không có data_path, '
                  f"nhận pose `[T,25,2]`; đổi toàn bộ Z sang inf cho chênh lệch xác suất {check['depth_difference']}. "
                  'Hook ở đầu model xác nhận `[1,3,64,25,2]` và toàn bộ kênh Z bằng 0. '
                  'Chi tiết: `models/fall_image2d_multiview/standalone_validation.json`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(root/'RESULTS.md')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
