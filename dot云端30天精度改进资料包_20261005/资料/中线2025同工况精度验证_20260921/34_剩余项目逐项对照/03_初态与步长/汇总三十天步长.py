"""核验相同初态、输入及参数的 15/7.5 秒完整30天无同化对照。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

O = Path(__file__).resolve().parent
R = O.parents[1]
B = R / '26_冻结基线与后续归因'
D15 = O.parent / '02_输入敏感性/M00_原输入完整30天'
D75 = O / 'I03_原初态7点5秒完整30天'
states = ('H_j', 'h_Ik', 'Q_ik', 'Q_uk', 'Q_dk', 'Q_o')
keys = ['station', 'time_s']
water = ['forebay_m', 'afterbay_m']
cfg15 = json.loads((D15 / '运行配置.json').read_text())
cfg75 = json.loads((D75 / '运行配置.json').read_text())
status15 = json.loads((D15 / '进度.json').read_text())
status75 = json.loads((D75 / '结果.json').read_text())
assert status15['status'] == status75['status'] == 'COMPLETED'
assert status15['completed_cycles'] == status75['periods'] == 1440
assert cfg75['dt'] == 7.5 and not cfg75['sync_gate_flow']
assert cfg15['inlet_delta_m3s'] == 0 and not cfg15['assimilation']
for key in ('input_sha256', 'initial_sha256'):
    assert cfg15[key] == cfg75[key]
assert hashlib.sha256((B / '共享输入/C07.json').read_bytes()).hexdigest() == cfg75['input_sha256']
assert hashlib.sha256((B / '共享输入/初始化后状态_0s.npz').read_bytes()).hexdigest() == cfg75['initial_sha256']
f15 = pd.read_csv(D15 / '半小时水位流量.csv')[keys + water].sort_values(keys).reset_index(drop=True)
f75 = pd.read_csv(D75 / '水位.csv')[keys + water].sort_values(keys).reset_index(drop=True)
for f in (f15, f75):
    assert len(f) == 1440 * 59 and not f.duplicated(keys).any()
    assert np.array_equal(np.sort(f.time_s.unique()), np.arange(1800, 2592001, 1800))
    assert np.isfinite(f[water].to_numpy()).all()
pd.testing.assert_frame_equal(f15[keys], f75[keys], check_exact=True)
prefix_checks = []
for full, old_name in ((f15, 'I00_原初态15秒'), (f75, 'I02_原初态7点5秒')):
    old = pd.read_csv(O / old_name / '水位.csv')[keys + water].sort_values(keys).reset_index(drop=True)
    prefix = full[full.time_s <= 259200].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, prefix, check_exact=True)
    prefix_checks.append({'source': old_name, 'rows': len(old), 'exact_equal': True})
final_diffs = {}
with np.load(D15 / '末态.npz') as a, np.load(D75 / '末态.npz') as b:
    for key in states:
        assert a[key].shape == b[key].shape
        assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
        final_diffs[key] = float(np.max(np.abs(a[key] - b[key])))
obs = pd.read_csv(B / '共享输入/作者已核实修正观测层.csv')
obs = obs[obs.valid & obs.metric.isin(water) & (obs.time_s > 0)]
obs = obs.rename(columns={'station_code': 'station', 'value': 'observed'})
series = {}
for name, f in (('15秒', f15), ('7.5秒', f75)):
    a = f.melt(id_vars=keys, value_vars=water, var_name='metric', value_name='sim')
    a = a.merge(obs[['station', 'time_s', 'metric', 'observed', 'pdf_comparable']], on=keys + ['metric'], validate='one_to_one')
    assert len(a) == 1440 * 118
    a['error'] = a.sim - a.observed
    series[name] = a
rows = []
periods = [('首3天', 0, 259200), ('后27天', 259200, 2592000), ('完整30天', 0, 2592000), ('末3天', 2332800, 2592000)]
for name, a in series.items():
    for label, start, end in periods:
        q = a[a.time_s.between(start, end, inclusive='right')]
        for scope, g in [('全线118个水位测点', q), ('作者可比88个水位测点', q[q.pdf_comparable])]:
            rows.append(dict(step=name, period=label, scope=scope, n=len(g), MAE=float(g.error.abs().mean()), RMSE=float(np.sqrt((g.error**2).mean())), max_abs=float(g.error.abs().max())))
scores = pd.DataFrame(rows)
scores.to_csv(O / '三十天步长精度对照.csv', index=False)
paired = series['7.5秒'].merge(series['15秒'][keys + ['metric', 'sim']], on=keys + ['metric'], suffixes=('', '_15'), validate='one_to_one')
paired['delta_m'] = paired.sim - paired.sim_15
diffs = []
for label, start, end in periods:
    q = paired[paired.time_s.between(start, end, inclusive='right')]
    peak = q.loc[q.delta_m.abs().idxmax()]
    diffs.append(dict(period=label, mean_abs_m=float(q.delta_m.abs().mean()), max_abs_m=float(q.delta_m.abs().max()), station=str(peak.station), metric=str(peak.metric), time_s=int(peak.time_s), signed_difference_m=float(peak.delta_m)))
pd.DataFrame(diffs).to_csv(O / '三十天步长水位差.csv', index=False)
result = dict(completed=True, cycles=1440, days=30, scored_samples_per_run=1440*118, same_input_and_initial_hash=True, first_three_days=prefix_checks, final_state_finite=True, final_state_max_difference=final_diffs, differences=diffs, run_15s=status15, run_7p5s=status75, note='无同化单成员时间步收敛核验；7.5秒不是实测真值。只检验当前几何、参数和已知历史边界，不证明物理建模正确。')
(O / '三十天步长核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
lines = ['# 完整30天时间步对照', '', '相同C07、共同初始化状态、原糙率和闸流系数，无同化单成员推进。两组均完成1440周期，覆盖2025年7月22日12时至8月21日12时。15秒组复用入口敏感性M00，不重复运行。', '', '## 与实测的水位误差', '', '|时间步|时段|测点范围|MAE/m|RMSE/m|最大绝对误差/m|', '|---|---|---|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['step']}|{r['period']}|{r['scope']}|{r['MAE']:.6f}|{r['RMSE']:.6f}|{r['max_abs']:.6f}|")
lines += ['', '## 两种时间步的水位差', '', '|时段|平均绝对差/mm|最大绝对差/mm|最大差位置|相对起点秒数|', '|---|---:|---:|---|---:|']
for d in diffs:
    point = d['station'] + ('闸前' if d['metric'] == 'forebay_m' else '闸后')
    lines.append(f"|{d['period']}|{d['mean_abs_m']*1000:.6f}|{d['max_abs_m']*1000:.6f}|{point}|{d['time_s']}|")
lines += ['', '## 核验和解释', '', '- 两组前3天分别与已完成的15秒、7.5秒短过程逐值精确相同；所有半小时输出和六类末态均有限。', '- 7.5秒是更小时间步的数值参照，不是实际水位真值；两种时间步接近不能消除原几何、阻力或输入偏差。', '- 接受步沿用strict=True、最大20次迭代；失败步按原规则回退减半，最低1.875秒，没有放宽门槛。', f"- 15秒组耗时{status15.get('wall_seconds', 0):.2f}秒，7.5秒组耗时{status75['wall_seconds']:.2f}秒；并发负载不同，这不是严格性能基准。", '- 这是历史边界回放，不是未知未来输入的独立预报。本轮没有修改或替换B01/B02。', '']
(O / '完整三十天步长对照报告.md').write_text('\n'.join(lines))
print(scores[scores.period == '完整30天'].to_string(index=False))
print(json.dumps(diffs, ensure_ascii=False, indent=2))
