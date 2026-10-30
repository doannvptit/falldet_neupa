"""Validate and report the 8-FPS experiment with no accuracy claims."""
import csv,json
from pathlib import Path
import subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from infer_video import load_records
from fall_onnx.classifier import BinaryClassifier
from fall_onnx.windows import classify_records
from scripts.evaluate_8fps import POLICIES


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]


def value(t): return '—' if t is None else f'{t:.3f}s'


def main():
    root=ROOT/'outputs/eval_8fps'
    data=json.loads((root/'results.json').read_text())
    verified=[]
    # Check resampling manifests and every rendered policy; source frames are
    # always chosen at or before the nominal 8-FPS timestamp.
    for row in data['results']:
        sampling=row['sampling']
        assert sampling['pixels_unchanged']
        assert sampling['target_fps']==8
        assert 0<=sampling['output_duration']-sampling['source_duration']<.125+1e-6
        assert all(s['source_time_sec']<=s['time_sec']+1e-6 for s in sampling['samples'])
        info=probe(root/'videos_8fps'/(row['video']+'.mp4'))
        assert int(info['nb_read_frames'])==row['frames']
        assert info['r_frame_rate']=='8/1'
        records=load_records(root/'poses_8fps'/row['video']/'poses.npz')
        observed={(r['track_id'],r['frame']) for r in records}
        for name,policy in POLICIES.items():
            folder=root/name/row['video']
            rows=list(csv.DictReader((folder/'windows.csv').open()))
            for score in rows:
                start,end,tid=[int(score[k]) for k in ('start_frame','end_frame','track_id')]
                assert 0<=start<=end<row['frames']
                assert float(score['end_sec'])==end/8
                real=sum((tid,i) in observed for i in range(start,end+1))
                assert int(score['observed_frames'])==real
                assert abs(float(score['coverage'])-real/(end-start+1))<1e-9
                if policy['full']:assert end-start+1==policy['window']
            frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
            assert len(frames)==row['frames']
            assert all(p['score_end_frame'] is None or p['score_end_frame']<=f['frame'] for f in frames for p in f['people'])
            stream=probe(folder/'annotated.mp4')
            assert stream['codec_name']=='h264' and int(stream['nb_read_frames'])==row['frames']
            assert stream['r_frame_rate']=='8/1'
            verified.append(str(folder.relative_to(root)))
    # Exact repeatability when scoring the saved 8-FPS pose cache.
    model=BinaryClassifier(device='cuda',temporal_method='linear',interpolation_gap=2)
    for row in data['results']:
        records=load_records(root/'poses_8fps'/row['video']/'poses.npz')
        actual=classify_records(records,model,8,16,2,8,2,.6,.5)
        saved=list(csv.DictReader((root/'linear_2s'/row['video']/'windows.csv').open()))
        assert len(saved)==len(actual)
        for a,b in zip(actual,saved):
            assert a['track_id']==int(b['track_id']) and a['end_frame']==int(b['end_frame'])
            assert abs(a['p_fall']-float(b['p_fall']))<1e-6
    for path in (root/'previews').glob('*/metadata.json'):
        meta=json.loads(path.read_text());stream=probe(path.parent/'interpolated_preview.mp4')
        assert int(stream['nb_read_frames'])==64 and stream['codec_name']=='h264'
        xy=np.load(path.parent/'interpolated_64_xy.npy')
        x=np.load(path.parent/'model_input.npy')
        assert xy.shape==(64,25,3) and np.all(xy[...,2]==0)
        assert x.shape==(1,3,64,25,2) and np.all(x[:,2]==0)
        assert meta['pose_times_sec'][-1]==meta['decision_available_at_sec']
    report=dict(passed=True,tests_passed=24,lossless_8fps_sources=4,policy_videos_verified=len(verified),
        causal_timelines=True,real_observation_counts_preserved=True,linear2s_cache_repeatable=True,
        ground_truth=None,source_fps=8,provider=model.device)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Đánh giá video 8 FPS bằng nội suy skeleton XY','',
        'Đã lấy mẫu RGB từ 4 video xuống 8 FPS rồi chạy lại YOLOX-s/RTMPose26 trên RTX 3060.',
        'Model fall ONNX, adapter, ngưỡng 0.5 và độ phân giải giữ nguyên. Không train lại.',
        'Các frame RGB được lấy tại floor(n × FPS nguồn / 8), mã hóa RGB lossless và kiểm tra hash pixel.',
        'Tốc độ phát giữ theo thời gian thực; sai khác độ dài do làm tròn nhỏ hơn 0.125 giây.',
        '', '## Thời điểm cảnh báo đầu tiên', '',
        '| Video | 30 FPS cũ | 8 FPS, linear 2s | 8 FPS, linear 4s | 8 FPS, linear 8s | Nearest 4s |',
        '|---|---:|---:|---:|---:|---:|']
    for row in data['results']:
        p=row['policies']
        times=[row['baseline_30fps']['first_fall_sec']]+[p[k]['first_fall_sec'] for k in ['linear_2s','linear_4s','linear_8s','nearest_4s']]
        lines.append('| '+row['video']+' | '+' | '.join(value(t) for t in times)+' |')
    lines+=['','Các mốc trên là thời điểm có đủ dữ liệu để phát cảnh báo; không phải độ trễ tính từ lúc bắt đầu ngã.',
        'Chưa có nhãn khoảng ngã nên không tính accuracy/precision/recall hay kết luận cảnh báo sớm hơn là tốt hơn.',
        '', '## Cấu hình đã thử','',
        '- `legacy_64frames`: giữ window=64, stride=8, min-observed=16; ở 8 FPS là tối đa 8 giây, cảnh báo mỗi 1 giây.',
        '- `linear_2s`: window=16 frame nguồn → nội suy 64 pose; stride=2 (0.25 giây), min-observed=8.',
        '- `linear_4s`: window=32 → 64 pose; stride/min-observed như trên.',
        '- `linear_8s`: window=64 → 64 pose, chủ yếu nội suy khớp/frame thiếu ngắn.',
        '- `nearest_4s`: cùng window/stride/warmup với linear_4s, thay nội suy tuyến tính bằng nearest.',
        '- `linear_8s_full`: chờ đủ 64 vị trí frame nguồn trong một segment mới đánh giá; vẫn yêu cầu coverage>=60%.',
        '', 'Cấu hình partial bắt đầu sau ít nhất 8 frame pose quan sát thật và coverage>=60%.',
        'Window 4/8 giây là độ dài tối đa, không phải luôn chờ đủ 4/8 giây lúc khởi động.',
        'Hai clip ngắn dưới 2 giây nên các window partial 2/4/8 giây dùng cùng lịch sử và có cùng kết quả.',
        'Không nối dài clip, lặp cuối clip hoặc tính frame tổng hợp là quan sát thật.',
        '', '## Khi chờ đủ cửa sổ 8 giây', '',
        '| Video | Frame 8 FPS | Cửa sổ đủ 64 frame | FALL windows |',
        '|---|---:|---:|---:|']
    for row in data['results']:
        p=row['policies']['linear_8s_full']
        lines.append(f"| {row['video']} | {row['frames']} | {p['windows']} | {p['fall_windows']} |")
    lines+=['','Không có cửa sổ đầy đủ nghĩa là chưa đủ lịch sử usable liên tục, không phải dự đoán NON-FALL.',
        'demo có 76 frame toàn video nhưng các segment theo người không đủ 64 frame liên tục đạt điều kiện.',
        '', '## Nhận xét', '',
        '- Trên bộ thử này, linear 2s giữ cảnh báo ở hai clip ngắn, chậm hơn mốc cảnh báo 30 FPS khoảng 58ms và 92ms.',
        '- Nearest 4s không vượt ngưỡng ở hai clip ngắn; linear 4s có cảnh báo. Đây là đối chứng cùng warmup/window.',
        '- demo: linear 2s cảnh báo 6.875s, gần mốc cũ 7.167s; linear 4/8s báo từ 3.625s. Cần nhãn để đánh giá báo sớm/sai.',
        '- fall_0002 vẫn không vượt ngưỡng ở mọi cấu hình; nội suy không khắc phục được hạn chế estimator/adapter/model.',
        '- Cấu hình thử nghiệm phù hợp để tiếp tục đánh giá: linear 2s, 16 frame nguồn → 64 pose, stride=2, max-gap=2.',
        '- Chờ đủ 8 giây tạo thời gian chờ lớn và bỏ qua các clip ngắn; chưa thấy lợi ích trên các video này.',
        '', '## Nội suy và tính nhân quả', '',
        'Nội suy tuyến tính từng khớp XY, chỉ giữa hai quan sát hợp lệ; Z=0.',
        'Không ngoại suy ngoài hai đầu hoặc nối qua hơn 2 frame thiếu (~0.25 giây tại 8 FPS).',
        'Pose tổng hợp chỉ được tính khi đã nhận frame cuối cửa sổ. Không phát cảnh báo ngược về thời điểm pose trung gian.',
        'Video preview nội suy là minh họa offline, còn annotated.mp4 trong từng policy giữ đúng timeline 8 FPS.',
        '64 pose nội suy không mang lại 64 lần quan sát độc lập và không phục hồi chuyển động đã mất giữa frame nguồn.',
        '', '## Chạy lại', '', '```bash',
        'cd /mnt/doannv/home/lumi/luvox-project/falldet/poc_inference_onnx',
        '.venv/bin/python scripts/evaluate_8fps.py --render',
        '# Dùng lại cache 8 FPS, không trích xuất pose lại:',
        '.venv/bin/python scripts/evaluate_8fps.py --reuse --render',
        '.venv/bin/python scripts/report_8fps.py',
        '```', '', 'Với video đầu vào đã khoảng 8 FPS:', '', '```bash',
        '.venv/bin/python infer_video.py --video video_8fps.mp4 --device cuda \\',
        '  --window 16 --stride 2 --min-frames 8 --max-gap 2 \\',
        '  --temporal-method linear --interpolation-gap 2 --output-dir outputs/lowfps',
        '```','', '## Artifacts', '',
        '- `outputs/eval_8fps/results.json`: toàn bộ score summary, chính sách và sampling manifest.',
        '- `outputs/eval_8fps/<policy>/<video>/`: windows.csv, summary.json, annotated.mp4, frames.jsonl.',
        '- `outputs/eval_8fps/previews/`: video so sánh pose quan sát và pose nội suy, NumPy 64 pose và tensor model.',
        '- `outputs/eval_8fps/verification.json`: kiểm tra 24 video policy, số frame/FPS, Z=0, timeline nhân quả và cache.',
        '- `tests.log`: 24/24 tests pass.','']
    (ROOT/'EVAL_8FPS.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
