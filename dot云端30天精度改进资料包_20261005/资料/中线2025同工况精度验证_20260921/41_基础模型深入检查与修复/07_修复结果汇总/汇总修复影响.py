"""41无同化修复对照，只读真实终态。geometry/numerical分支独立汇总，不合并修复。"""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R41 = HERE.parent
R = R41.parent
WORKSPACE = R.parents[1]
B = R / '26_冻结基线与后续归因'
G00 = R / '38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数'
TRACE00 = R / '39_局部与全线边界归因/03_接口与水量补录/G00_冻结原参数'
GEO = R41 / '05_西赵河无同化验证'
NUM = R41 / '06_闸流水量一致性修复/G00_更严闸流一致性'
TIMES = np.arange(1800, 2592001, 1800)
PERIODS = [('全30天', 0, 2592000), ('后15天', 1296000, 2592000)]
STATE_KEYS = ['H_j', 'h_Ik', 'Q_ik', 'Q_uk', 'Q_dk', 'Q_o']


def path_of(value):
    p = Path(value)
    return p if p.is_absolute() else WORKSPACE / p


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(None)
def sha(path):
    return hashlib.sha256(path_of(path).read_bytes()).hexdigest()


def arrsha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def fingerprints(cfg):
    require(all(sha(p) == h for p, h in cfg['source_hashes'].items()), '运行来源指纹改变')
    for module in cfg['runtime_identity']['modules']:
        if module['name'].startswith(('wxq_pipedream', 'pipedream_solver')):
            p = Path(module.get('file', module.get('path'))).resolve()
            require(p.is_relative_to((B / '共享依赖').resolve()), '加载了非26水力实现')
            require(sha(p) == module['sha256'], '水力模块指纹改变')


def terminal(folder, numeric=False):
    p = folder / ('结果.json' if numeric else '进度.json')
    r = read(p) if p.exists() else {'status': 'NOT_STARTED'}
    done = r['status'] in ['COMPLETED', 'FAILED']
    gate = folder / '修复区输入轮廓门槛停止.json'
    return dict(folder=str(folder), terminal=done, status=r['status'],
                completed_cycles=r.get('completed_cycles', 0), error=r.get('error'),
                geometry_gate=read(gate) if gate.exists() else None,
                classification='INPUT_CONTOUR_PREREQUISITE_FAILED' if gate.exists() else r['status'],
                terminal_file_sha256=sha(p) if p.exists() else None)


def complete(status):
    return status['status'] == 'COMPLETED' and status['completed_cycles'] == 1440


def check_grid(frame, point_keys, expected_points, cycles=1440):
    require(len(frame) == cycles * expected_points, f'行数不是{cycles}×{expected_points}')
    require(not frame.duplicated(['time_s', *point_keys]).any(), '时刻/测点重复')
    expected_times = np.arange(1800, (cycles + 1) * 1800, 1800)
    require(np.array_equal(np.sort(frame.time_s.unique()), expected_times), '共同半小时时间轴不完整')
    require((frame.groupby('time_s').size() == expected_points).all(), '某时刻测点数不足')
    require((frame.groupby(point_keys).size() == cycles).all(), '某测点时间样本不完整')


def observations():
    obs = pd.read_csv(B / '共享输入/作者已核实修正观测层.csv')
    require(not obs.duplicated(['station_code', 'metric', 'time_s']).any(), '冻结观测主键重复')
    return obs.rename(columns={'station_code': 'station', 'value': 'observed'})


def forcing_sources():
    mapping = pd.read_csv(R / '02_输入/2025节制闸站名映射.csv').set_index('station_code')
    model = read(B / '共享输入/C07.json')['中线全线']
    targets = model['timeSeries']['seriesLink'].values()
    require(['nodes', 'NODE-main-0'] in targets and ['nodes', 'NODE-main-119'] in targets, '两外边界绑定变了')
    bound = [('NODE-main-0', str(mapping.loc['BOUNDARY_IN', 'source_name']), 'flow_m3s'),
             ('NODE-main-119', str(mapping.loc['BOUNDARY_OUT', 'source_name']), 'forebay_m')]
    return mapping, bound


def attach_observed(frame, obs):
    frame = frame.merge(obs[['station', 'metric', 'time_s', 'observed', 'valid', 'pdf_comparable', 'sheet']],
                        on=['station', 'metric', 'time_s'], how='left', validate='many_to_one')
    require(frame.valid.notna().all() and frame.valid.all(), '当前冻结观测缺失/无效，禁止补值或改掩码')
    require(np.isfinite(frame[['sim', 'observed']]).all().all(), '模拟或观测非有限')
    mapping, bound = forcing_sources()
    frame['source_name'] = frame.station.map(mapping.source_name)
    require(frame.source_name.notna().all() and (frame.source_name == frame.sheet).all(), '评分列源站身份不符')
    frame['direct_forcing_observation'] = [any(name == source and metric == value for _, source, value in bound)
                                            for name, metric in zip(frame.source_name, frame.metric)]
    frame['error'] = frame.sim - frame.observed
    return frame


def load_scores(folder, case, obs, meters=None):
    node = pd.read_csv(folder / '半小时水位流量.csv')
    mapped = pd.read_csv(folder / '实际位置水位.csv')
    check_grid(node, ['station'], 59)
    check_grid(mapped, ['station', 'metric'], 43)
    require(np.isfinite(node[['forebay_m', 'afterbay_m', 'gate_flow_m3s']]).all().all(), '节点结果非有限')
    require(np.isfinite(mapped.mapped_sim).all(), '位置水位非有限')
    long_node = node.melt(id_vars=['time_s', 'station'], value_vars=['forebay_m', 'afterbay_m'], var_name='metric', value_name='sim')
    long_node['kind'], long_node['heldout'], long_node['chainage_m'] = 'nodes', False, np.nan
    mapped = mapped.rename(columns={'mapped_sim': 'sim'})
    mapped['kind'], mapped['heldout'] = 'mapped', mapped.group == '旧留出检验测点'
    frozen_ops = read(B / 'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json')
    frozen_flags = {(x['station'], x['metric']): not x['assimilated'] for x in frozen_ops}
    actual_flags = {(x.station, x.metric): x.heldout for x in mapped[['station', 'metric', 'heldout']].drop_duplicates().itertuples()}
    require(actual_flags == frozen_flags, '43点CSV分组不等于冻结assimilated名单')
    require(mapped[['station', 'metric', 'heldout']].drop_duplicates().heldout.sum() == 9, '旧9留出身份不符')
    parts = [long_node, mapped]
    if meters is not None:
        require(set(meters.metric) == {'flow_m3s'}, '测流评分混入其他量')
        check_grid(meters, ['station'], meters.station.nunique())
        meters['kind'], meters['heldout'] = 'meters', False
        parts.append(meters)
    frame = pd.concat(parts, ignore_index=True)
    frame['case'] = case
    frame = attach_observed(frame, obs)
    frame['end_connection_approximation'] = (frame.kind == 'meters') & frame.station.isin(['JZZ17', 'JZZ59'])
    frame['end_connection_offset_m'] = np.where(frame.end_connection_approximation,
                                                frame.station.map({'JZZ17': 33., 'JZZ59': -40.}), np.nan)
    return frame


def groups(frame, include_additional_groups=False):
    mapped = frame.kind == 'mapped'
    meters = frame[frame.kind == 'meters']
    result = {'118闸节点水位': frame[frame.kind == 'nodes'], '43安装位置水位': frame[mapped],
              '旧9留出水位': frame[mapped & frame.heldout]}
    if not meters.empty:
        result['44测流算子_描述性' if meters.station.nunique() == 44 else 'JZZ16测流断面_半小时末态'] = meters
    if include_additional_groups:
        result['原88可比节点水位'] = frame[(frame.kind == 'nodes') & frame.pdf_comparable]
        result['旧更新34水位'] = frame[mapped & ~frame.heldout]
    return result


def numerical_group_membership(obs):
    water = obs[obs.metric.isin(['forebay_m', 'afterbay_m'])]
    require((water.groupby(['station', 'metric']).pdf_comparable.nunique() == 1).all(), '原可比标志随时刻变化')
    keys88 = set(map(tuple, water[water.pdf_comparable][['station', 'metric']].drop_duplicates().to_numpy()))
    historical_path = R / '02_输入/仅作者88侧初始化观测.csv'
    historical = pd.read_csv(historical_path)
    historical = historical[historical.metric.isin(['forebay_m', 'afterbay_m']) & historical.valid]
    historical88 = set(map(tuple, historical[['station_code', 'metric']].drop_duplicates().to_numpy()))
    require(len(keys88) == 88 and keys88 == historical88, '冻结pdf_comparable与历史明确88名单不一致')
    ops_path = B / 'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json'
    ops = read(ops_path)
    updated = [(x['station'], x['metric']) for x in ops if x['assimilated']]
    require(len(ops) == 43 and len(updated) == len(set(updated)) == 34, '冻结34更新名单不完整')
    return dict(definitions={'原88可比节点水位': '26冻结观测pdf_comparable=True，并与02历史仅作者88侧初始化的有效水位名单逐点一致。',
                             '旧更新34水位': '26冻结43观测算子assimilated=True；实际CSV分组须逐点匹配该标志。'},
                original88_points=[dict(station=s, metric=m) for s, m in sorted(keys88)],
                updated34_points=[dict(station=s, metric=m) for s, m in updated],
                selection_depends_on_error=False,
                source_sha256={str(p): sha(p) for p in [historical_path, ops_path, B / '共享输入/作者已核实修正观测层.csv']})


def stats(values):
    e = np.asarray(values, float)
    require(len(e) > 0 and np.isfinite(e).all(), '不能给空结果或非有限值评分')
    return dict(samples=len(e), MAE=float(abs(e).mean()), RMSE=float(np.sqrt(np.mean(e**2))),
                bias=float(e.mean()), max_abs=float(abs(e).max()))


def score_tables(frames, include_additional_groups=False):
    overall, pointwise = [], []
    for frame in frames:
        for period, lo, hi in PERIODS:
            use = frame[(frame.time_s > lo) & (frame.time_s <= hi)]
            for group, part in groups(use, include_additional_groups).items():
                if part.empty:
                    continue
                base = dict(case=str(part.case.iloc[0]), period=period, score_group=group,
                            unit='m3/s' if (part.metric == 'flow_m3s').all() else 'm')
                overall.append(dict(**base, points=len(part[['station', 'metric']].drop_duplicates()), **stats(part.error)))
                for (station, metric), rows in part.groupby(['station', 'metric']):
                    x = rows.iloc[0]
                    pointwise.append(dict(**base, station=station, metric=metric, **stats(rows.error),
                                          source_name=x.source_name, direct_forcing_observation=bool(x.direct_forcing_observation),
                                          end_connection_approximation=bool(x.end_connection_approximation),
                                          end_connection_offset_m=None if pd.isna(x.end_connection_offset_m) else float(x.end_connection_offset_m)))
    return pd.DataFrame(overall), pd.DataFrame(pointwise)


def changes_table(points, baseline, candidate):
    keys = ['period', 'score_group', 'unit', 'station', 'metric', 'samples']
    if points.empty or candidate not in set(points.case):
        empty = pd.DataFrame(columns=[*keys, 'delta_MAE', 'delta_RMSE', 'delta_bias', 'delta_max_abs'])
        empty['water_MAE_worse_over_1cm'] = pd.Series(dtype=bool)
        return empty
    old = points[points.case == baseline]
    new = points[points.case == candidate]
    paired = new.merge(old, on=keys, suffixes=('_candidate', '_reference'), validate='one_to_one')
    require(len(paired) == len(new) == len(old), '候选与对照评分测点不完全配对')
    for key in ['MAE', 'RMSE', 'bias', 'max_abs']:
        paired['delta_' + key] = paired[key + '_candidate'] - paired[key + '_reference']
    paired['water_MAE_worse_over_1cm'] = (paired.unit == 'm') & (paired.delta_MAE > .01)
    return paired


def global_water(folder, case, cfg):
    a = pd.read_csv(folder / '全网半小时水量账.csv')
    require(len(a) == 1440 and np.array_equal(a.start_s, np.arange(1440) * 1800) and np.array_equal(a.end_s, TIMES), '全网水量账时轴不完整')
    numeric = a.select_dtypes(include='number')
    require(np.isfinite(numeric).all().all() and (a.accepted_steps >= 120).all(), '全网水量账数值或成功步数异常')
    require(np.max(abs(a.discrete_balance_residual_m3 - (a.discrete_continuity_storage_change_m3 - a.net_boundary_input_m3))) < 1e-7, '离散水量账算术不符')
    require(np.max(abs(a.geometry_minus_discrete_change_m3 - (a.geometric_storage_change_m3 - a.discrete_continuity_storage_change_m3))) < 1e-7, '几何-离散蓄量差算术不符')
    rows = []
    for period, lo, hi in PERIODS:
        z = a[(a.start_s >= lo) & (a.end_s <= hi)]
        require(len(z) == (hi - lo) // 1800, '水量时段缺行')
        rows.append(dict(case=case, period=period, control_volume='全网内部连接与非定水位主节点', duration_s=hi-lo,
                         net_boundary_input_m3=float(z.net_boundary_input_m3.sum()),
                         native_storage_change_m3=float(z.discrete_continuity_storage_change_m3.sum()),
                         native_minus_net_input_residual_m3=float(z.discrete_balance_residual_m3.sum()),
                         max_successful_step_discrete_residual_m3s=float(z.maximum_step_discrete_residual_m3s.max()),
                         geometric_storage_change_m3=float(z.geometric_storage_change_m3.sum()),
                         geometry_minus_native_change_m3=float(z.geometry_minus_discrete_change_m3.sum()),
                         max_halfhour_abs_geometry_minus_native_m3=float(z.geometry_minus_discrete_change_m3.abs().max()),
                         accepted_steps=int(z.accepted_steps.sum()),
                         branch_t0_geometric_volume_m3=cfg['candidate_initial_volume_m3'],
                         window_start_geometric_volume_m3=cfg['candidate_initial_volume_m3']+float(a.loc[a.end_s <= lo, 'geometric_storage_change_m3'].sum()),
                         window_end_geometric_volume_m3=cfg['candidate_initial_volume_m3']+float(a.loc[a.end_s <= hi, 'geometric_storage_change_m3'].sum()),
                         interpretation='两种模型储水口径之差及离散方程残差，不是现场水损；每分支从自身初始蓄量记账'))
    return rows


def geometry_metadata(configs):
    old = read(G00 / '运行配置.json')
    ref, new = configs['reference_full'], configs['western_repair']
    for cfg in configs.values():
        fingerprints(cfg)
        require(cfg['parameters_fixed'] and cfg['internal_observation_updates'] == 0 and not cfg['baseline_replaced'], '改参数/同化/替换状态不符')
    require(ref['initial_state_sha256'] == old['initial_state_sha256'] and ref['fixed_parameter_sha256'] == old['parameter_sha256'], '原模型身份不等于G00')
    require(ref['forcing_array_sha256'] == old['forcing_array_sha256'], '原模型forcing不等于G00')
    for key in ['times', 'h', 'q', 'u']:
        require(new['forcing_array_sha256'][key] == ref['forcing_array_sha256'][key], '边界或开度不同: ' + key)
    for key in ['_n_uk', '_n_dk', '_Co', '_C_uk', '_C_dk']:
        require(new['fixed_parameter_sha256'][key] == ref['fixed_parameter_sha256'][key], '非网格相关参数不同: ' + key)
    for key in ['H_j', 'Q_uk', 'Q_dk', 'Q_o']:
        require(new['initial_state_sha256'][key] == ref['initial_state_sha256'][key], '应保留的初态数组不同: ' + key)
    require(not new['initial_six_states_identical'] and new['geometry_changed'], '几何初态被误称六状态相同')
    proof = read(R41 / '03_结构转换核查/西赵河初态投影独立核查.json')
    require(proof['source_npz_sha256'] == sha(B / '共享输入/初始化后状态_0s.npz'), '投影证据来源初态不符')
    head_rows = [x for x in new['projection'] if 'absolute_head_m' in x]
    require(len(head_rows) == 2, '两个端点投影记录不完整')
    with np.load(B / '共享输入/初始化后状态_0s.npz') as z:
        for evidence in proof['special_endpoints']:
            row = next(x for x in head_rows if x['node'] == evidence['source_node'])
            require(row['x'] == evidence['physical_position_m'] and row['channel'] == evidence['channel'], '端点身份投影错误')
            require(abs(row['absolute_head_m'] - z['H_j'][evidence['source_node_id']]) < 1e-12 and
                    abs(row['new_depth_m'] - evidence['new_depth_m']) < 1e-12, '投影不是原共享节点水头')
    inputs = {}
    for case in configs:
        inputs[case] = read(GEO / case / '分水逐物理节点一致性.json')
        q = inputs[case]
        require(q['all_shared_values_exact'] and q['removed_nodes_all_zero'] and q['added_nodes_all_zero'] and q['sum_roundoff_max_m3s'] < 1e-10, '物理分水输入不同')
    return dict(source_hashes_verified=True, same_boundaries_and_openings=True, same_physical_withdrawals=inputs,
                two_shared_node_head_projections=head_rows, six_states_identical=False,
                original_initial_volume_m3=ref['original_initial_volume_m3'], candidate_initial_volume_m3=new['candidate_initial_volume_m3'],
                geometric_initial_volume_difference_m3=new['candidate_initial_volume_m3']-ref['original_initial_volume_m3'],
                parameter_note='候选新增网格使_n_ik数组hash不同；各superlink的n值保持由冻结运行器断言核对，其余五个参数数组hash相同。')


def save_tables(prefix, frames, baseline, candidate, include_additional_groups=False):
    overall, points = score_tables(frames, include_additional_groups)
    paired = changes_table(points, baseline, candidate)
    overall.to_csv(HERE / (prefix + '_分组精度.csv'), index=False)
    points.to_csv(HERE / (prefix + '_逐测点精度.csv'), index=False)
    paired.to_csv(HERE / (prefix + '_逐点增减.csv'), index=False)
    paired[paired.water_MAE_worse_over_1cm].to_csv(HERE / (prefix + '_水位MAE退化超过1cm.csv'), index=False)
    return overall, paired


def summarize_geometry(check_only):
    statuses = {name: terminal(GEO / name) for name in ['reference_full', 'western_repair']}
    if not all(x['terminal'] for x in statuses.values()):
        return dict(status='WAITING_FOR_TRUE_TERMINALS', branch='geometry', cases=statuses, formal_metrics_written=False)
    configs = {name: read(GEO / name / '运行配置.json') for name in statuses}
    identity = geometry_metadata(configs)
    require(complete(statuses['reference_full']), '完整原模型对照尚未成立')
    node = pd.read_csv(GEO / 'reference_full/半小时水位流量.csv')
    require(node.equals(pd.read_csv(G00 / '半小时水位流量.csv')), '原对照84960行未逐值复现G00')
    with np.load(GEO / 'reference_full/末态.npz') as a, np.load(G00 / '末态.npz') as b:
        require(all(np.array_equal(a[k], b[k]) for k in STATE_KEYS), '原对照六末态未逐值复现G00')
    reference_ops = read(GEO / 'reference_full/实际位置观测算子.json')
    candidate_ops = read(GEO / 'western_repair/实际位置观测算子.json')
    semantic = ['station', 'metric', 'chainage', 'assimilated', 'channel', 'state']
    require([{k: x[k] for k in semantic} for x in reference_ops] == [{k: x[k] for k in semantic} for x in candidate_ops], '43实际位置语义不一致')
    obs, frames, water, meter_scope = observations(), [], [], []
    for case, state in statuses.items():
        if not complete(state):
            continue  # 不读取“未完成”CSV计算部分候选收益。
        folder = GEO / case
        q = pd.read_csv(folder / '测流位置算子流量.csv').rename(columns={'model_flow_m3s': 'sim'})
        check_grid(q, ['station'], 44)
        q['metric'] = 'flow_m3s'
        frame = load_scores(folder, case, obs, q)
        frames.append(frame)
        water += global_water(folder, case, configs[case])
        scope = frame[frame.kind == 'meters'][['case', 'station', 'chainage_m', 'source_name', 'direct_forcing_observation',
                                                'end_connection_approximation', 'end_connection_offset_m']].drop_duplicates()
        meter_scope += scope.to_dict('records')
        ranges = pd.read_csv(folder / '修复区半小时轮廓范围.csv')
        require(np.array_equal(np.sort(ranges.time_s.unique()), TIMES), '轮廓记录时间不完整')
        if case == 'western_repair':
            require(ranges.excess_m.max() <= 1e-7 and state['geometry_gate'] is None, '候选轮廓前提未通过')
    eligible = complete(statuses['western_repair'])
    result = dict(status='COMPLETED_PAIRED_DESCRIPTION' if eligible else 'CANDIDATE_PREREQUISITE_FAILED_NO_FULL_MONTH_GAIN',
                  branch='geometry', cases=statuses, identity=identity, reference_full_exact_G00=True,
                  candidate_full_month_eligible=eligible, partial_candidate_scored=False,
                  interpretation='候选若到达未核实输入轮廓范围，表示当前几何顶部假设不满足；不等于数值不收敛或现场漫顶。不得抬高轮廓或把已完成部分时段当全月收益。',
                  forcing_source_columns=forcing_sources()[1], source_hashes={str(GEO / case / '运行配置.json'): sha(GEO / case / '运行配置.json') for case in statuses},
                  baseline_replaced=False, engineering_acceptance=False)
    result['artifact_sha256'] = {str(p): sha(p) for case in statuses for p in (GEO / case).iterdir()
                                 if p.suffix in ['.json', '.csv', '.npz']}
    result['endpoint_meter_note'] = 'JZZ17最近连接中心比请求位置下游33m；JZZ59上游40m。44点全部描述性列出，不称全部严格同坐标或新独立盲测。'
    result['direct_forcing_flow_validation_points'] = sorted({x['station'] for x in meter_scope if x['direct_forcing_observation']})
    if not check_only:
        _, paired = save_tables('西赵河几何修复', frames, 'reference_full', 'western_repair')
        pd.DataFrame(water).to_csv(HERE / '西赵河几何修复_水量与储水口径.csv', index=False)
        pd.DataFrame(meter_scope).to_csv(HERE / '测流算子输入来源与端部近似.csv', index=False)
        result['degraded_water_points_over_1cm'] = paired[paired.water_MAE_worse_over_1cm].to_dict('records')
        result['formal_metrics_written'] = True
        result['formal_metrics_cases'] = [name for name, state in statuses.items() if complete(state)]
        result['candidate_scores_generated'] = eligible
        dump(HERE / '西赵河几何修复结论.json', result)
    return result


def read_control_trace(files, case, node_count=None):
    end, steps, snapshots, interval_sums = 0., 0, [], []
    sums = {p: dict(duration_s=0., meter_volume_m3=0., gate_volume_m3=0., offtake_volume_m3=0.,
                   native_storage_change_m3=0., integrated_cv_residual_m3=0., max_cv_residual_m3s=0.,
                   cv_steps_over_1e6=0, max_nonboundary_node_residual_m3s=None,
                   max_successful_iterations=None) for p, _, _ in PERIODS}
    for file, expected_hash in files:
        require(expected_hash is None or sha(file) == expected_hash, '逐步分块hash不符')
        with np.load(path_of(file)) as z:
            t, start, dt = z['time_s'], z['interval_start_s'], z['dt_s']
            require(start[0] == end and np.array_equal(start[1:], t[:-1]) and np.array_equal(t-start, dt) and (dt > 0).all(), '成功子步时间轴不连续')
            cv = z['Q16_cut_m3s']-z['Q16_gate_m3s']-z['offtake_volume_m3']/dt-z['native_storage_change_m3']/dt
            require(np.max(abs(cv-z['continuity_residual_m3s'])) < 1e-10, '控制体残差算术不符')
            if 'nonboundary_node_residual_m3s' in z:
                nr = z['nonboundary_node_residual_m3s']
                require(np.isfinite(nr).all() and nr.shape[0] == len(t), '节点残差分块错位')
                require(node_count is None or nr.shape[1] == node_count, '节点残差列数与节点索引不符')
                nrmax = abs(nr).max(axis=1)
                require(np.max(abs(nrmax-z['max_nonboundary_node_residual_m3s'])) < 1e-12, '节点最大残差不符')
            else:
                require(node_count is None, '数值修复分块缺少节点残差')
                nrmax = None
            for period, lo, hi in PERIODS:
                mask = (start >= lo) & (t <= hi)
                if not mask.any():
                    continue
                a = sums[period]
                a['duration_s'] += float(dt[mask].sum())
                for key, values in [('meter_volume_m3', z['Q16_cut_m3s']*dt), ('gate_volume_m3', z['Q16_gate_m3s']*dt),
                                    ('offtake_volume_m3', z['offtake_volume_m3']), ('native_storage_change_m3', z['native_storage_change_m3']),
                                    ('integrated_cv_residual_m3', cv*dt)]:
                    a[key] += float(values[mask].sum())
                a['max_cv_residual_m3s'] = max(a['max_cv_residual_m3s'], float(abs(cv[mask]).max()))
                a['cv_steps_over_1e6'] += int((abs(cv[mask]) > 1e-6).sum())
                if nrmax is not None:
                    a['max_nonboundary_node_residual_m3s'] = max(a['max_nonboundary_node_residual_m3s'] or 0., float(nrmax[mask].max()))
                if 'iterations' in z:
                    a['max_successful_iterations'] = max(a['max_successful_iterations'] or 0, int(z['iterations'][mask].max()))
            half = t % 1800 == 0
            snapshots.append(pd.DataFrame(dict(time_s=t[half], station='JZZ16', metric='flow_m3s',
                                               chainage_m=300091., sim=z['Q16_cut_m3s'][half])))
            interval_sums.append(pd.DataFrame(dict(time_s=np.ceil(t / 1800)*1800,
                                duration_s=dt, meter_m3=z['Q16_cut_m3s']*dt)).groupby('time_s').sum())
            end, steps = float(t[-1]), steps + len(t)
    require(end == 2592000 and sums['全30天']['duration_s'] == 2592000 and sums['后15天']['duration_s'] == 1296000, '数值分支逐步覆盖不足30天')
    rows = [dict(case=case, period=p, control_volume='兰河300091m测流切口至JZZ16过闸断面', **v,
                 sign='净入控制体流量减离散蓄变；与全网账的储水减净输入符号相反，不合并',
                 geometric_storage_difference='此分支未记录几何-离散储水差，不猜造') for p, v in sums.items()]
    mean = pd.concat(interval_sums).groupby(level=0).sum().reset_index()
    require(np.array_equal(mean.time_s, TIMES) and (mean.duration_s == 1800).all(), '半小时积分不是完整1440段')
    mean['model_mean_m3s'] = mean.meter_m3 / mean.duration_s
    return pd.concat(snapshots, ignore_index=True), rows, steps, mean


def summarize_numerical(check_only):
    status = terminal(NUM, numeric=True)
    if not status['terminal']:
        return dict(status='WAITING_FOR_TRUE_TERMINAL', branch='numerical_only', case=status, formal_metrics_written=False)
    if not complete(status):
        result = dict(status='NUMERICAL_BRANCH_FAILED_NO_FULL_MONTH_GAIN', branch='numerical_only', case=status,
                      partial_candidate_scored=False, baseline_replaced=False)
        if not check_only:
            dump(HERE / '闸流数值一致性修复结论.json', result)
        return result
    cfg, old = read(NUM / '运行配置.json'), read(G00 / '运行配置.json')
    fingerprints(cfg)
    require(cfg['protocol']['flow_tol_m3s'] == 1e-9 and cfg['protocol']['head_tol_m'] == .0015, '不是单独flow_tol修复')
    require(not cfg['assimilation'] and cfg['state_corrections'] == 0, '数值分支有同化')
    for key, old_key in [('initial_state_sha256', 'initial_state_sha256'), ('parameter_sha256', 'parameter_sha256'), ('forcing_array_sha256', 'forcing_array_sha256')]:
        require(cfg[key] == old[old_key], '数值分支改变初态/参数/边界')
    with np.load(NUM / '实际初始化_0s.npz') as z:
        require(all(arrsha(z[k]) == cfg['initial_state_sha256'][k] for k in STATE_KEYS), '数值分支实际初态指纹不符')
    index = read(NUM / '逐成功子步文件索引.json')
    require(index['scalar_and_node_rows_aligned'], '数值分块行未对齐')
    require(sum(x['rows'] for x in index['chunks']) == index['total_rows'], '数值分块总行数不符')
    node_metadata = read(NUM / index['nonboundary_node_metadata'])
    require(len(node_metadata['ids']) == len(node_metadata['names']) == node_metadata['count'] and
            len(set(node_metadata['ids'])) == node_metadata['count'], '非定水位节点索引不完整')
    candidate_q, candidate_water, count, candidate_mean = read_control_trace([(x['file'], x['sha256']) for x in index['chunks']], 'flow_tol_1e-9', node_metadata['count'])
    require(count == index['total_rows'], '数值分块实际行数不符')
    reference_q, reference_water, _, reference_mean = read_control_trace([(TRACE00 / '逐成功子步接口及水量.npz', None)], 'G00_original')
    obs = observations()
    frames = [load_scores(G00, 'G00_original', obs, reference_q), load_scores(NUM, 'flow_tol_1e-9', obs, candidate_q)]
    membership = numerical_group_membership(obs)
    for frame in frames:
        selected_groups = groups(frame, include_additional_groups=True)
        for name, count in [('原88可比节点水位', 88), ('旧更新34水位', 34)]:
            require(len(selected_groups[name]) == 1440 * count, name + '时点覆盖不完整')
    saved = read(NUM / '结果.json')
    require(candidate_water[0]['max_successful_iterations'] == saved['maximum_iterations'], '成功步最大迭代数不符')
    require(saved['all_successful_steps_control_volume_gate_pass'] == (candidate_water[0]['cv_steps_over_1e6'] == 0), '数值控制体门槛标记不符')
    require(saved['all_successful_steps_all_nonboundary_nodes_under_1e6'] ==
            (candidate_water[0]['max_nonboundary_node_residual_m3s'] <= 1e-6), '全部主节点门槛标记不符')
    require(saved['old_39_failure_unchanged'] and not saved['baseline_replaced'], '改写旧门槛或替换状态')
    rejected = [json.loads(line) for line in (NUM / '减步失败尝试.jsonl').read_text().splitlines() if line.strip()]
    require(len(rejected) == saved['rejected_attempts'] == saved['retries'], '失败尝试未完整保留')
    require(all(x['six_state_rollback_exact'] and x['rolled_back_time_s'] == x['start_s'] for x in rejected), '失败步回退不一致')
    raw_q = obs[(obs.station == 'JZZ16') & (obs.metric == 'flow_m3s')].sort_values('time_s')
    require(np.array_equal(raw_q.time_s, np.arange(1441)*1800), 'Q16实测均流所需端点不完整')
    observed_mean = (raw_q.observed.to_numpy()[:-1]+raw_q.observed.to_numpy()[1:])/2
    mean_scores = []
    for case, folder, means in [('G00_original', TRACE00, reference_mean), ('flow_tol_1e-9', NUM, candidate_mean)]:
        ledger = pd.read_csv(folder / '半小时测流至闸水量账.csv')
        require(np.array_equal(ledger.time_s, means.time_s) and np.max(abs(ledger.model_meter_mean_m3s-means.model_mean_m3s)) < 1e-10, '水量账均流与实际dt积分不符')
        for period, lo, hi in PERIODS:
            use = (means.time_s > lo) & (means.time_s <= hi)
            mean_scores.append(dict(case=case, period=period, quantity='JZZ16测流断面_半小时均流', unit='m3/s',
                                    **stats(means.loc[use, 'model_mean_m3s']-observed_mean[use]),
                                    basis='模型按成功子步Q×dt积分；观测以相邻半小时原值线性时间均值，不当作新增高频实测。'))
    result = dict(status='COMPLETED_INDEPENDENT_NUMERICAL_DESCRIPTION', branch='numerical_only', case=status,
                  only_change='flow_tol=1e-9，原几何/原n/C/原初态/原输入，无同化，不与western_repair合并',
                  successful_substeps=count, all_steps_control_volume_gate_pass=saved['all_successful_steps_control_volume_gate_pass'],
                  all_nonboundary_node_gate_pass=saved['all_successful_steps_all_nonboundary_nodes_under_1e6'],
                  water_balance=candidate_water + reference_water,
                  source_hashes={**cfg['source_hashes'], **membership['source_sha256'], str(Path(__file__)): sha(Path(__file__))},
                  score_group_membership=membership,
                  Q16_halfhour_mean_scores=mean_scores,
                  iteration_counts=dict(successful_step_max=candidate_water[0]['max_successful_iterations'],
                    successful_step_distribution=saved['iteration_counts'], rejected_attempts=len(rejected),
                    rejected_attempt_iteration_counts=[x['iterations'] for x in rejected],
                    all_rejected_six_state_and_time_rollbacks_exact=True,
                    note='成功步与被拒绝尝试分开；不能以成功步最多4次说全部尝试最多4次。'),
                  baseline_replaced=False, engineering_acceptance=False,
                  limitations='仅44测流中的JZZ16有同位置逐步读数，其余43测流没有本分支输出，不补造；收紧数值一致性不等于精度或工程通过。')
    result['artifact_sha256'] = {str(p): sha(p) for p in NUM.iterdir() if p.suffix in ['.json', '.csv', '.npz']}
    result['chunk_sha256'] = {x['file']: x['sha256'] for x in index['chunks']}
    if not check_only:
        overall, paired = save_tables('闸流数值一致性修复', frames, 'G00_original', 'flow_tol_1e-9', include_additional_groups=True)
        result['group_accuracy'] = overall.to_dict('records')
        keys = ['period', 'score_group', 'unit', 'points', 'samples']
        group_delta = overall[overall.case == 'flow_tol_1e-9'].merge(overall[overall.case == 'G00_original'],
                            on=keys, suffixes=('_candidate', '_reference'), validate='one_to_one')
        for metric in ['MAE', 'RMSE', 'bias', 'max_abs']:
            group_delta['delta_' + metric] = group_delta[metric + '_candidate']-group_delta[metric + '_reference']
        result['group_accuracy_changes'] = group_delta.to_dict('records')
        result['largest_point_MAE_increase_by_group'] = [rows.loc[rows.delta_MAE.idxmax()].to_dict()
                         for _, rows in paired.groupby(['period', 'score_group'])]
        pd.DataFrame(mean_scores).to_csv(HERE / '闸流数值一致性修复_Q16半小时均流误差.csv', index=False)
        pd.DataFrame(candidate_water + reference_water).to_csv(HERE / '闸流数值一致性修复_控制体水量账.csv', index=False)
        result['degraded_water_points_over_1cm'] = paired[paired.water_MAE_worse_over_1cm].to_dict('records')
        result['formal_metrics_written'] = True
        dump(HERE / '闸流数值一致性修复结论.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=['geometry', 'numerical', 'all'], default='geometry')
    parser.add_argument('--check-only', action='store_true', help='只读核验真实终态，不写任何正式汇总')
    args = parser.parse_args()
    results = []
    for name, function in [('geometry', summarize_geometry), ('numerical', summarize_numerical)]:
        if args.branch not in [name, 'all']:
            continue
        try:
            results.append(function(args.check_only))
        except Exception as exc:
            error = dict(status='BLOCKED_BY_ARTIFACT_OR_IDENTITY_CHECK', branch=name, error=repr(exc), formal_metrics_written=False)
            if not args.check_only:
                dump(HERE / (name + '_汇总阻断.json'), error)
            results.append(error)
    if not args.check_only:
        reports = [HERE / '西赵河几何修复结论.json', HERE / '闸流数值一致性修复结论.json']
        available = [dict(path=str(p), sha256=sha(p), result=read(p)) for p in reports if p.exists()]
        ready = len(available) == 2 and all(not x['status'].startswith(('WAITING', 'BLOCKED')) for x in results)
        dump(HERE / '基础模型修复汇总索引.json', dict(status='BOTH_INDEPENDENT_BRANCHES_REPORTED' if ready else 'WAITING_FOR_OTHER_BRANCH',
             independent_branches=available, this_invocation=results, geometry_and_numerical_combined=False,
             baseline_replaced=False, engineering_acceptance=False,
             note='几何候选若轮廓前提失败，不给部分月份收益；两个分支分别评价，不混成已修复基础模型。全网求和小残差不能替代局部控制体或逐节点门槛。'))
    print(json.dumps([dict(branch=x['branch'], status=x['status'], error=x.get('error')) for x in results], ensure_ascii=False, indent=2))
    return 2 if any(x['status'].startswith('BLOCKED') for x in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
