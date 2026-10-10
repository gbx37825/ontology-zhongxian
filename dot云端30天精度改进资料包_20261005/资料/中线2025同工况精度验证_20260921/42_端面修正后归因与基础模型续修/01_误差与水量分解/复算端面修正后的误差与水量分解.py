"""只读41/10完整配对：误差符号、闸两侧位移、沿程落差及全网水量差。

不导入水力模块，不推进模型，不选择新的验收子集，不修改41或26。
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / '41_基础模型深入检查与修复/10_端面几何锁参验证'
FROZEN = ROOT / '26_冻结基线与后续归因'
CASES = ['reference', 'endpoint']
TIMES = np.arange(1800, 2592001, 1800)
PERIODS = [('全30天', 0, 2592000), ('前15天', 0, 1296000), ('后15天', 1296000, 2592000)]
SOURCES = {}
OUTPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracked(path):
    path = Path(path).resolve()
    SOURCES[str(path)] = sha(path)
    return path


def read_json(path):
    return json.loads(tracked(path).read_text())


def read_csv(path):
    return pd.read_csv(tracked(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def save_csv(name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    path = HERE / name
    frame.to_csv(path, index=False)
    OUTPUTS[name] = dict(rows=len(frame), sha256=sha(path))


def metrics(g):
    e0, e1, delta = [g[x].to_numpy(float) for x in ['error_reference', 'error_endpoint', 'delta_H_m']]
    require(np.isfinite(np.c_[e0, e1, delta]).all(), '水位或误差含非有限值')
    require(np.max(abs(e1-e0-delta)) < 1e-10, '误差变化不是同一观测下的模型变化')
    linear = np.sign(e0)*delta
    crossing = abs(e1)-abs(e0)-linear
    require(crossing.min() >= -1e-10, '绝对误差符号分解不符')
    result = dict(samples=len(g), reference_bias_m=float(e0.mean()), endpoint_bias_m=float(e1.mean()),
                  reference_MAE_m=float(abs(e0).mean()), endpoint_MAE_m=float(abs(e1).mean()),
                  delta_MAE_m=float(abs(e1).mean()-abs(e0).mean()),
                  reference_RMSE_m=float(np.sqrt(np.mean(e0**2))), endpoint_RMSE_m=float(np.sqrt(np.mean(e1**2))),
                  reference_max_abs_m=float(abs(e0).max()), endpoint_max_abs_m=float(abs(e1).max()),
                  mean_delta_H_m=float(delta.mean()), min_delta_H_m=float(delta.min()), max_delta_H_m=float(delta.max()),
                  reference_negative_error_fraction=float((e0 < 0).mean()),
                  reference_positive_error_fraction=float((e0 > 0).mean()),
                  sign_crossing_samples=int((e0*e1 < 0).sum()),
                  reference_sign_times_delta_H_m=float(linear.mean()), crossing_correction_m=float(crossing.mean()))
    require(abs(result['delta_MAE_m']-result['reference_sign_times_delta_H_m']-result['crossing_correction_m']) < 1e-12,
            'MAE分解均值恒等式不符')
    if 'common_delta_H_m' in g:
        result['mean_common_delta_H_m'] = float(g.common_delta_H_m.mean())
        result['mean_gate_difference_component_m'] = float(g.gate_difference_component_m.mean())
        result['MAE_linear_common_component_m'] = float((np.sign(e0)*g.common_delta_H_m).mean())
        result['MAE_linear_gate_difference_component_m'] = float((np.sign(e0)*g.gate_difference_component_m).mean())
        require(abs(result['MAE_linear_common_component_m']+result['MAE_linear_gate_difference_component_m']
                    + result['crossing_correction_m']-result['delta_MAE_m']) < 1e-12, '共同/差分水位的MAE分解不符')
    if 'delta_H_same_station_node_m' in g:
        result['mean_delta_H_same_station_node_m'] = float(g.delta_H_same_station_node_m.mean())
        result['mean_mapped_minus_node_response_m'] = float((g.delta_H_m-g.delta_H_same_station_node_m).mean())
    return result


def validate_grid(frame, keys, points):
    require(len(frame) == 1440*points and not frame.duplicated(['time_s', *keys]).any(), '完整输出行数/主键不符')
    require(np.array_equal(np.sort(frame.time_s.unique()), TIMES), '半小时时间轴不完整')
    require((frame.groupby(keys).size() == 1440).all(), '测点时间覆盖不足')


def build_water_level_tables(obs):
    a, b = [read_csv(RUNS / case / '半小时水位流量.csv') for case in CASES]
    for frame in [a, b]:
        validate_grid(frame, ['station'], 59)
    wide = a.merge(b, on=['time_s', 'station'], suffixes=('_reference', '_endpoint'), validate='one_to_one')
    wide['station_number'] = wide.station.str[3:].astype(int)
    require(set(wide.station_number) == set(range(2, 61)), '本分区定义要求原59座JZZ2至JZZ60')
    observed = obs[obs.metric.isin(['forebay_m', 'afterbay_m'])].pivot(index=['time_s', 'station_code'], columns='metric', values='value')
    observed = observed.rename(columns={'forebay_m': 'observed_forebay_m', 'afterbay_m': 'observed_afterbay_m'}).reset_index().rename(columns={'station_code': 'station'})
    wide = wide.merge(observed, on=['time_s', 'station'], validate='one_to_one')
    for side in ['forebay_m', 'afterbay_m']:
        wide['delta_' + side] = wide[side+'_endpoint']-wide[side+'_reference']
        for case in CASES:
            wide['error_'+case+'_'+side] = wide[side+'_'+case]-wide['observed_'+side]
    wide['common_delta_H_m'] = (wide.delta_forebay_m+wide.delta_afterbay_m)/2
    wide['gate_level_difference_delta_m'] = wide.delta_forebay_m-wide.delta_afterbay_m
    wide['gate_flow_delta_m3s'] = wide.gate_flow_m3s_endpoint-wide.gate_flow_m3s_reference
    wide['observed_gate_level_difference_m'] = wide.observed_forebay_m-wide.observed_afterbay_m
    for case in CASES:
        wide['gate_level_difference_'+case+'_m'] = wide['forebay_m_'+case]-wide['afterbay_m_'+case]
        wide['gate_level_difference_error_'+case+'_m'] = wide['gate_level_difference_'+case+'_m']-wide.observed_gate_level_difference_m
    parts = []
    for metric, sign in [('forebay_m', 1.), ('afterbay_m', -1.)]:
        g = wide[['time_s', 'station', 'station_number', 'common_delta_H_m']].copy()
        g['scope'], g['metric'] = '闸节点表示', metric
        g['error_reference'] = wide['error_reference_'+metric]
        g['error_endpoint'] = wide['error_endpoint_'+metric]
        g['delta_H_m'] = wide['delta_'+metric]
        g['gate_difference_component_m'] = sign*wide.gate_level_difference_delta_m/2
        require(np.max(abs(g.delta_H_m-g.common_delta_H_m-g.gate_difference_component_m)) < 1e-12, '闸两侧位移分解不符')
        parts.append(g)
    node = pd.concat(parts, ignore_index=True)
    mapped = [read_csv(RUNS/case/'实际位置水位.csv') for case in CASES]
    for g in mapped:
        validate_grid(g, ['station', 'metric'], 43)
    m = mapped[0].merge(mapped[1], on=['time_s', 'station', 'metric', 'group', 'chainage_m'], suffixes=('_reference', '_endpoint'), validate='one_to_one')
    m = m.merge(obs[['station_code', 'time_s', 'metric', 'value']].rename(columns={'station_code': 'station'}), on=['station', 'time_s', 'metric'], validate='one_to_one')
    m['scope'] = '既有43安装位置'
    m['error_reference'], m['error_endpoint'] = m.mapped_sim_reference-m.value, m.mapped_sim_endpoint-m.value
    m['delta_H_m'] = m.mapped_sim_endpoint-m.mapped_sim_reference
    m = m.merge(node[['station', 'metric', 'time_s', 'delta_H_m']].rename(columns={'delta_H_m': 'delta_H_same_station_node_m'}),
                on=['station', 'metric', 'time_s'], validate='one_to_one')
    rows = []
    for frame in [node, m]:
        for period, lo, hi in PERIODS:
            for (station, metric), g in frame[(frame.time_s > lo) & (frame.time_s <= hi)].groupby(['station', 'metric']):
                rows.append(dict(period=period, scope=str(g.scope.iloc[0]), station=station, metric=metric, **metrics(g)))
    save_csv('分期水位误差与符号分解.csv', rows)
    return wide, node, m, pd.DataFrame(rows)


def station_stats(g):
    row = dict(samples=len(g), mean_forebay_delta_m=float(g.delta_forebay_m.mean()),
               mean_afterbay_delta_m=float(g.delta_afterbay_m.mean()), common_delta_H_m=float(g.common_delta_H_m.mean()),
               gate_level_difference_delta_m=float(g.gate_level_difference_delta_m.mean()),
               gate_flow_reference_mean_m3s=float(g.gate_flow_m3s_reference.mean()),
               gate_flow_endpoint_mean_m3s=float(g.gate_flow_m3s_endpoint.mean()), gate_flow_delta_mean_m3s=float(g.gate_flow_delta_m3s.mean()))
    for side in ['forebay_m', 'afterbay_m']:
        a, b = g['error_reference_'+side], g['error_endpoint_'+side]
        row.update({side+'_reference_bias_m': float(a.mean()), side+'_endpoint_bias_m': float(b.mean()),
                    side+'_reference_MAE_m': float(abs(a).mean()), side+'_endpoint_MAE_m': float(abs(b).mean())})
    for case in CASES:
        e = g['gate_level_difference_error_'+case+'_m']
        row['gate_level_difference_'+case+'_mean_m'] = float(g['gate_level_difference_'+case+'_m'].mean())
        row['gate_level_difference_'+case+'_bias_m'] = float(e.mean())
        row['gate_level_difference_'+case+'_MAE_m'] = float(abs(e).mean())
    return row


def regional_and_station_tables(wide, node):
    regions = [('JZZ2至46', 2, 46), ('JZZ47至55', 47, 55), ('JZZ56至60', 56, 60), ('全线118节点', 2, 60)]
    region_rows, station_rows, daily = [], [], []
    for period, lo, hi in PERIODS:
        frame = node[(node.time_s > lo) & (node.time_s <= hi)]
        for name, lower, upper in regions:
            g = frame[frame.station_number.between(lower, upper)]
            count = (upper-lower+1)*2
            result = metrics(g)
            region_rows.append(dict(period=period, region=name, points=count, **result,
                                    weighted_contribution_to_global_delta_MAE_m=result['delta_MAE_m']*count/118))
        for station, g in wide[(wide.time_s > lo) & (wide.time_s <= hi)].groupby('station'):
            station_rows.append(dict(period=period, station=station, **station_stats(g)))
    regions_frame = pd.DataFrame(region_rows)
    for period, _, _ in PERIODS:
        g = regions_frame[regions_frame.period == period]
        require(abs(g[g.region != '全线118节点'].weighted_contribution_to_global_delta_MAE_m.sum()-
                    g.loc[g.region == '全线118节点', 'delta_MAE_m'].iloc[0]) < 1e-12, '分区加权与全线MAE变化不一致')
    selected = wide[wide.station_number.between(47, 55)].copy()
    selected['day'] = np.ceil(selected.time_s/86400).astype(int)
    for (day, station), g in selected.groupby(['day', 'station']):
        daily.append(dict(model_day=int(day), start_exclusive_s=(int(day)-1)*86400,
                          end_inclusive_s=int(day)*86400, station=station, **station_stats(g)))
    save_csv('分区加权误差贡献.csv', regions_frame)
    save_csv('闸站共同位移与闸前后水位差.csv', station_rows)
    save_csv('关注区逐日演变.csv', daily)
    return regions_frame, pd.DataFrame(station_rows), pd.DataFrame(daily)


def interstation_and_telescope(wide):
    by_station = {int(g.station_number.iloc[0]): g.sort_values('time_s').reset_index(drop=True) for _, g in wide.groupby('station')}
    tables, raw = [], {}
    for station in range(2, 60):
        up, down = by_station[station], by_station[station+1]
        r = up.afterbay_m_reference-down.forebay_m_reference
        e = up.afterbay_m_endpoint-down.forebay_m_endpoint
        raw[station] = e-r
        for period, lo, hi in PERIODS:
            use = (up.time_s > lo) & (up.time_s <= hi)
            tables.append(dict(period=period, upstream_station=f'JZZ{station}', downstream_station=f'JZZ{station+1}',
                               reference_level_drop_mean_m=float(r[use].mean()), endpoint_level_drop_mean_m=float(e[use].mean()),
                               delta_level_drop_mean_m=float((e-r)[use].mean()),
                               meaning='相邻闸后至下座闸前的节点水位落差，未含速度水头，不等于区间总能量损失。'))
    anchor = by_station[55].delta_afterbay_m.to_numpy()
    gate_sum = sum(by_station[i].gate_level_difference_delta_m.to_numpy() for i in range(48, 56))
    reach_sum = sum(raw[i].to_numpy() for i in range(48, 55))
    require(np.max(abs(by_station[48].delta_forebay_m-anchor-gate_sum-reach_sum)) < 1e-10, '沿程水位差望远镜恒等式不符')
    decomposition = []
    for period, lo, hi in PERIODS:
        use = (by_station[48].time_s > lo) & (by_station[48].time_s <= hi)
        decomposition.append(dict(period=period, JZZ48_forebay_delta_mean_m=float(by_station[48].delta_forebay_m[use].mean()),
             JZZ55_afterbay_anchor_delta_mean_m=float(anchor[use].mean()),
             sum_JZZ48_to55_gate_level_difference_delta_m=float(gate_sum[use].mean()),
             sum_interstation_JZZ48_to55_level_drop_delta_m=float(reach_sum[use].mean()),
             interpretation='精确的节点水位差恒等式，仅定位分布变化，不判定变化起源或区间内部损失构成。'))
    save_csv('相邻闸间模型水位落差变化.csv', tables)
    return decomposition


def flow_table(observations):
    frames = [read_csv(RUNS/case/'测流位置算子流量.csv') for case in CASES]
    for frame in frames:
        validate_grid(frame, ['station'], 44)
    merged = frames[0].merge(frames[1], on=['time_s', 'station', 'chainage_m'], suffixes=('_reference', '_endpoint'), validate='one_to_one')
    measured = observations[observations.metric == 'flow_m3s'][['station_code', 'time_s', 'value']].rename(columns={'station_code': 'station'})
    merged = merged.merge(measured, on=['station', 'time_s'], validate='one_to_one')
    merged['delta_flow_m3s'] = merged.model_flow_m3s_endpoint-merged.model_flow_m3s_reference
    rows = []
    for period, lo, hi in PERIODS:
        for station, g in merged[(merged.time_s > lo) & (merged.time_s <= hi)].groupby('station'):
            er, ee = g.model_flow_m3s_reference-g.value, g.model_flow_m3s_endpoint-g.value
            rows.append(dict(period=period, station=station, chainage_m=float(g.chainage_m.iloc[0]), samples=len(g),
                 reference_model_mean_m3s=float(g.model_flow_m3s_reference.mean()), endpoint_model_mean_m3s=float(g.model_flow_m3s_endpoint.mean()),
                 mean_delta_flow_m3s=float(g.delta_flow_m3s.mean()), min_delta_flow_m3s=float(g.delta_flow_m3s.min()), max_delta_flow_m3s=float(g.delta_flow_m3s.max()),
                 reference_bias_m3s=float(er.mean()), endpoint_bias_m3s=float(ee.mean()),
                 reference_MAE_m3s=float(abs(er).mean()), endpoint_MAE_m3s=float(abs(ee).mean()),
                 delta_MAE_m3s=float(abs(ee).mean()-abs(er).mean()),
                 reference_positive_error_fraction=float((er > 0).mean()),
                 end_connection_offset_m={'JZZ17': 33., 'JZZ59': -40.}.get(station),
                 temporal_basis='半小时末态瞬时值序列的样本均值，不作为Q×dt积分替代。'))
    save_csv('44测流分期响应.csv', rows)
    return merged, pd.DataFrame(rows)


def prescribed_forcing_integrals(source):
    arrays = {(kind, name): np.asarray(source['timeSeries']['seriesData'][key], float)
              for key, (kind, name) in source['timeSeries']['seriesLink'].items()}
    require(sum(kind == 'nodes' for kind, _ in arrays) == 2 and sum(kind == 'bleeders' for kind, _ in arrays) == 128,
            '规定输入范围不是已核验两边界/128分水')
    require([name for name, n in source['baseData']['nodes'].items() if int(n.get('nodeType', 0)) == 1] == ['NODE-main-119'],
            '不能将净边界差统一解释为单个定水位尾端净通量')
    times = arrays['nodes', 'NODE-main-0'][:, 0]
    net = arrays['nodes', 'NODE-main-0'][:, 1]-sum(v[:, 1] for (kind, _), v in arrays.items() if kind == 'bleeders')
    tail = arrays['nodes', 'NODE-main-119'][:, 1]
    output = {}
    for case in CASES:
        index = read_json(RUNS/case/'逐成功子步文件索引.json')
        pieces, snapshots, previous_end, steps = [], [], 0., 0
        for chunk in index['chunks']:
            p = tracked(chunk['file'])
            require(sha(p) == chunk['sha256'], '成功步分块指纹改变')
            with np.load(p) as z:
                start, end, dt = z['interval_start_s'], z['time_s'], z['dt_s']
                require(start[0] == previous_end and np.array_equal(start[1:], end[:-1]) and np.array_equal(end-start, dt), '成功步时间轴不连续')
                pieces.append(pd.DataFrame(dict(time_s=np.ceil(end/1800)*1800, prescribed_net_volume_m3=np.interp(start, times, net)*dt,
                                                duration_s=dt)).groupby('time_s').sum())
                mask = end % 1800 == 0
                snapshots.append(pd.DataFrame(dict(time_s=end[mask], last_step_start_s=start[mask], last_dt_s=dt[mask],
                                    applied_tail_stage_m=np.interp(start[mask], times, tail))))
                previous_end, steps = float(end[-1]), steps+len(end)
        totals = pd.concat(pieces).groupby(level=0).sum().reset_index()
        snap = pd.concat(snapshots, ignore_index=True)
        require(previous_end == 2592000 and steps == index['total_rows'] and np.array_equal(totals.time_s, TIMES)
                and (totals.duration_s == 1800).all() and np.array_equal(snap.time_s, TIMES), '实际输入积分覆盖不足')
        output[case] = totals.merge(snap, on='time_s', validate='one_to_one')
    return output


def water_budget(configs, forcing, wide):
    a, b = [read_csv(RUNS/case/'全网半小时水量账.csv') for case in CASES]
    require(len(a) == len(b) == 1440 and np.array_equal(a.end_s, TIMES) and np.array_equal(b.end_s, TIMES), '全网账时间轴不完整')
    out = pd.DataFrame(dict(start_s=a.start_s, time_s=a.end_s))
    for field in ['net_boundary_input_m3', 'discrete_continuity_storage_change_m3', 'geometric_storage_change_m3',
                  'discrete_balance_residual_m3', 'geometry_minus_discrete_change_m3']:
        out['delta_'+field] = b[field]-a[field]
    out['delta_prescribed_net_volume_m3'] = forcing['endpoint'].prescribed_net_volume_m3-forcing['reference'].prescribed_net_volume_m3
    out['delta_inferred_tail_net_outward_volume_m3'] = out.delta_prescribed_net_volume_m3-out.delta_net_boundary_input_m3
    out['delta_inferred_tail_net_outward_mean_m3s'] = out.delta_inferred_tail_net_outward_volume_m3/1800
    out['delta_applied_tail_stage_at_output_m'] = forcing['endpoint'].applied_tail_stage_m-forcing['reference'].applied_tail_stage_m
    initial_delta = configs['endpoint']['initial_geometric_volume_m3']-configs['reference']['initial_geometric_volume_m3']
    require(initial_delta == 0., '两组初始计量体积不同，必须另列而非隐去')
    out['cumulative_delta_native_storage_changes_m3'] = out.delta_discrete_continuity_storage_change_m3.cumsum()
    out['cumulative_delta_geometric_volume_m3'] = initial_delta+out.delta_geometric_storage_change_m3.cumsum()
    out['cumulative_delta_net_boundary_input_m3'] = out.delta_net_boundary_input_m3.cumsum()
    out['cumulative_delta_geometry_minus_native_m3'] = initial_delta+out.delta_geometry_minus_discrete_change_m3.cumsum()
    require(np.max(abs(out.delta_discrete_continuity_storage_change_m3-out.delta_net_boundary_input_m3-out.delta_discrete_balance_residual_m3)) < 1e-7,
            '配对差额的离散水量恒等式不符')
    save_csv('全网半小时差额与边界核验.csv', out)
    periods = []
    for period, lo, hi in PERIODS:
        g = out[(out.start_s >= lo) & (out.time_s <= hi)]
        periods.append(dict(period=period, duration_s=hi-lo,
            delta_net_boundary_input_m3=float(g.delta_net_boundary_input_m3.sum()),
            delta_native_storage_changes_m3=float(g.delta_discrete_continuity_storage_change_m3.sum()),
            delta_geometric_storage_changes_m3=float(g.delta_geometric_storage_change_m3.sum()),
            delta_geometry_minus_native_m3=float(g.delta_geometry_minus_discrete_change_m3.sum()),
            delta_prescribed_net_input_reconstructed_m3=float(g.delta_prescribed_net_volume_m3.sum()),
            delta_inferred_tail_net_outward_volume_m3=float(g.delta_inferred_tail_net_outward_volume_m3.sum()),
            delta_inferred_tail_net_outward_mean_m3s=float(g.delta_inferred_tail_net_outward_volume_m3.sum()/(hi-lo)),
            delta_discrete_balance_residual_m3=float(g.delta_discrete_balance_residual_m3.sum()),
            max_abs_applied_tail_stage_difference_m=float(g.delta_applied_tail_stage_at_output_m.abs().max())))
    return out, periods


def temporal_tables(wide, node, water, flow):
    rows = []
    for end in range(21600, 2592001, 21600):
        lo = end-21600
        w = water[(water.start_s >= lo) & (water.time_s <= end)]
        x = wide[(wide.time_s > lo) & (wide.time_s <= end) & (wide.station == 'JZZ48')]
        whole = node[(node.time_s > lo) & (node.time_s <= end)]
        north = whole[whole.station_number.between(47, 55)]
        rows.append(dict(start_exclusive_s=lo, end_inclusive_s=end, **station_stats(x),
            global118_mean_delta_H_m=float(whole.delta_H_m.mean()), northern18_mean_delta_H_m=float(north.delta_H_m.mean()),
            delta_net_boundary_input_m3=float(w.delta_net_boundary_input_m3.sum()),
            delta_native_storage_changes_m3=float(w.delta_discrete_continuity_storage_change_m3.sum()),
            delta_geometric_storage_changes_m3=float(w.delta_geometric_storage_change_m3.sum()),
            cumulative_delta_native_storage_changes_m3=float(w.cumulative_delta_native_storage_changes_m3.iloc[-1]),
            cumulative_delta_geometric_volume_m3=float(w.cumulative_delta_geometric_volume_m3.iloc[-1]),
            delta_tail_net_outward_mean_m3s=float(w.delta_inferred_tail_net_outward_volume_m3.sum()/21600)))
    save_csv('固定六小时水量与JZZ48响应.csv', rows)
    key = wide[wide.station == 'JZZ48'].copy()
    key = key.merge(flow[flow.station == 'JZZ48'][['time_s', 'delta_flow_m3s']], on='time_s', validate='one_to_one')
    columns = ['time_s', 'delta_forebay_m', 'delta_afterbay_m', 'common_delta_H_m', 'gate_level_difference_delta_m',
               'gate_flow_delta_m3s', 'delta_flow_m3s', 'error_reference_forebay_m', 'error_endpoint_forebay_m',
               'error_reference_afterbay_m', 'error_endpoint_afterbay_m', 'gate_level_difference_error_reference_m',
               'gate_level_difference_error_endpoint_m']
    save_csv('JZZ48半小时诊断序列.csv', key[columns])
    onset = []
    for (station, metric), g in node[node.station_number.between(47, 55)].groupby(['station', 'metric']):
        g = g.sort_values('time_s')
        values, times = g.delta_H_m.to_numpy(), g.time_s.to_numpy()
        for threshold in [1e-6, .001, .005, .01]:
            def first(mask):
                positions = np.flatnonzero(mask)
                return None if not len(positions) else float(times[positions[0]])
            negative = values < -threshold
            sustained = np.convolve(negative.astype(int), np.ones(7, dtype=int), mode='valid') == 7
            positions = np.flatnonzero(sustained)
            onset.append(dict(station=station, metric=metric, diagnostic_threshold_m=threshold,
                first_abs_above_threshold_s=first(abs(values) > threshold), first_negative_below_threshold_s=first(negative),
                first_positive_above_threshold_s=first(values > threshold),
                first_3h_sustained_negative_start_s=None if not len(positions) else float(times[positions[0]]),
                minimum_delta_H_m=float(values.min()), minimum_time_s=float(times[values.argmin()]),
                resolution_note='首次只能定位到半小时保存时刻；连续3h为7个端点，不据此推断传播速度或因果起源。'))
    save_csv('关注区变化起始与峰值定位.csv', onset)
    return pd.DataFrame(rows), pd.DataFrame(onset)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    configs = {}
    for case in CASES:
        result = read_json(RUNS/case/'结果.json')
        require(result['status'] == 'COMPLETED' and result['full_30day_completed'] and result['completed_cycles'] == 1440,
                '本归因只接受完整30天真实终态')
        configs[case] = read_json(RUNS/case/'运行配置.json')
        require(not configs[case]['assimilation'] and configs[case]['state_corrections'] == 0, '本轮不允许同化')
    for key in ['initial_state_sha256', 'parameter_sha256', 'forcing_array_sha256', 'internal_geometry_sha256']:
        require(configs['reference'][key] == configs['endpoint'][key], '配对基础身份不同: ' + key)
    for cfg in configs.values():
        for path, digest in cfg['source_hashes'].items():
            p = Path(path)
            p = p if p.is_absolute() else ROOT.parents[1]/p
            require(sha(tracked(p)) == digest, '运行来源指纹已变化: ' + str(p))
    observation_path = FROZEN/'共享输入/作者已核实修正观测层.csv'
    obs = read_csv(observation_path)
    require(not obs.duplicated(['station_code', 'metric', 'time_s']).any() and obs.valid.all(), '冻结观测主键/有效标志变化')
    source = read_json(FROZEN/'共享输入/C07.json')['中线全线']
    for filename in ['实际位置观测算子.json', '测流位置算子与适用范围.json']:
        require(read_json(RUNS/'reference'/filename) == read_json(RUNS/'endpoint'/filename), '算子不同: ' + filename)
    wide, node, mapped, point_stats = build_water_level_tables(obs)
    region_stats, station_table, daily = regional_and_station_tables(wide, node)
    telescope = interstation_and_telescope(wide)
    flow, flow_stats = flow_table(obs)
    forcing = prescribed_forcing_integrals(source)
    water, water_periods = water_budget(configs, forcing, wide)
    sixhour, onset = temporal_tables(wide, node, water, flow)
    known_summary = read_json(RUNS/'端面几何锁参验证结论.json')
    for period in ['全30天', '后15天']:
        g = region_stats[(region_stats.period == period) & (region_stats.region == '全线118节点')].iloc[0]
        for case in CASES:
            previous = next(x for x in known_summary['group_accuracy'] if x['period'] == period and x['case'] == case and x['score_group'] == '118闸节点水位')
            require(abs(previous['MAE']-g[case+'_MAE_m']) < 1e-12, '独立重算与原118评分不符')
    j48 = station_table[station_table.station == 'JZZ48']
    j48_fore = point_stats[(point_stats.scope == '闸节点表示') & (point_stats.station == 'JZZ48') & (point_stats.metric == 'forebay_m')]
    top_daily = daily[daily.station == 'JZZ48'].sort_values('mean_forebay_delta_m').head(3)
    northern_mapped = mapped[mapped.station.str[3:].astype(int).between(47, 55)][['station', 'metric']].drop_duplicates()
    independent_candidates = []
    for filename in [ROOT/'42_端面修正后归因与基础模型续修/03_剩余方程一致性/梅河闸底原表独立复核.json',
                     ROOT/'42_端面修正后归因与基础模型续修/02_退化渠段结构核查/梅河首断面构模与算子差分.json']:
        if filename.exists():
            independent_candidates.append(dict(evidence_path=str(tracked(filename)),
                 relevance='独立原表/转换核查的新候选依据，未在本脚本运行或评分，不能称已解释JZZ47–55退化。'))
    tracked(Path(__file__))
    result = dict(status='READ_ONLY_ATTRIBUTION_COMPLETED', checked_at_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='41/10完整30天的事后诊断，不新跑水动力、不调n/C、不启用同化、不改变验收集。',
        definitions=dict(error='model - frozen_observation', delta='endpoint - reference',
            shared_level_component='c=(deltaH_fore+deltaH_after)/2', level_difference_component='d=deltaH_fore-deltaH_after',
            forebay_identity='deltaH_fore=c+d/2', afterbay_identity='deltaH_after=c-d/2',
            MAE_identity='delta_MAE = mean(sign(e_reference)*deltaH) + mean(abs(e_endpoint)-abs(e_reference)-sign(e_reference)*deltaH)',
            head_difference='这里是节点水位差，未含流速水头，不直接称总能量损失。',
            global_level='118节点等权平均，不是体积加权水位。', temporal='日/6h区间按模型秒数划分；不是重选有利检验时段。'),
        regional_decomposition=records(region_stats), JZZ48_station_decomposition=records(j48),
        JZZ48_forebay_error_sign_decomposition=records(j48_fore), JZZ48_downstream_anchor_identity=telescope,
        JZZ48_onset=records(onset[(onset.station == 'JZZ48') & (onset.metric == 'forebay_m')]),
        most_negative_JZZ48_forebay_response_days_posthoc=records(top_daily),
        water_difference=water_periods,
        interpretation=[
            '全线MAE是空间加权结果。JZZ2–46、JZZ47–55、JZZ56–60三段穷尽原118节点，并验证加权变化精确相加；该分区仅用于归因。',
            '同向水位下降对原本偏低的47–55增加绝对误差，对原本多数偏高的56–60反而减小误差；原误差符号不能省略。',
            'JZZ48后15天原闸前误差全负，新增下降量精确对应MAE增加。共同位移与前后水位差分量是代数分解，不是已识别的两项独立物理因果。',
            '规定入流/分水/水位时序相同，但两组真实减步时间网格不同；重构规定净输入积分差与模型净尾端通量响应分开。',
            '全网离散蓄变差与净边界差闭合，不代表额外体积位于JZZ47–55。几何与离散储水差不能当现场水损。'],
        location_qualification=dict(northern_existing_position_points=records(northern_mapped),
            JZZ48_forebay_has_saved_position_operator=bool(((mapped.station == 'JZZ48') & (mapped.metric == 'forebay_m')).any()),
            note='JZZ48前水位不在既有43安装位置算子中；节点表示的退化仍按预定规则保留，不能因位置未闭合而删分，也不能直接断言现场同位置误差如此变化。'),
        excluded_explanations=[
            '本配对中上游/分水时序被暗改、n/C重新率定、内部实测更新或初始六态不同：配置指纹逐项相同。',
            '只是全线水位统一平移：三个区域的平均deltaH方向不同，且同站前后差也改变。',
            '全线平均改善意味着每个北段测点改善：逐点与误差符号分解直接反驳。',
            '将全网几何-离散储水差或尾端模型通量差直接当作现场漏水/补水。'],
        mechanisms_to_test=[
            dict(id='M1_LOCAL_VS_NETWORK', hypothesis='北段局部端面变化与上游/末段变化引起的全网回水及流量重分配共同作用，现有两组全量hook不能分配各自因果份额。',
                 proposed_test='按已核实拓扑预先冻结JZZ47–55相邻开渠端面集合，做仅本集合与仅其余端面的锁参机制消融；先覆盖第一条显著响应的起算短段，不把短段变成新验收集。',
                 required_records='局部两端H/Q、端面A/B/Pe/R、动量各项及对应渠池体积；相同t0/forcing/容差，无同化。'),
            dict(id='M2_REMAINING_HEAD_FLOW_RELATION', hypothesis='北段水位原本偏低而部分测流原本偏高；降低流量与降低共同水位可能改善一项、损害另一项。剩余能量损失/边界代表性及仪器位置尚不能由节点差值区分。',
                 proposed_test='对已定位的关键闸间区段建立含流速水头和结构损失的同源能量账，并核实JZZ48前实际仪器位置/高程；先定位缺项，不用水位误差反调n/C或补造输入。',
                 required_records='同一断面的速度/过水面积、闸前后及邻段瞬时H/Q、原表结构参数与仪器身份。')],
        independent_new_candidates_not_attributed=independent_candidates,
        limits=['本轮没有逐段完整内场时序或能量损失分项；端点/闸间水位差不定位到某段摩阻或某个局部损失。',
                '早起变化受半小时输出分辨率限制，不据首次超阈时间推传播速度。',
                '固定6h/日演变和响应排序仅事后定位，不变成新的保留集或选参窗。',
                '梅河四孔闸底115.521/116.723与q20s1桩号385036/385306的新线索来自独立源资料核查，未证明其解释北段退化或带来精度收益。'],
        original_conditional_accuracy_gate_unchanged=known_summary['conditional_accuracy_gate'],
        baseline_replaced=False, engineering_acceptance=False, simulations_run=0,
        output_files=OUTPUTS, source_sha256=SOURCES)
    require(all(sha(path) == digest for path, digest in SOURCES.items()), '只读分析期间来源改变')
    output = HERE/'端面修正后误差与水量归因.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], output=str(output), CSV_files=len(OUTPUTS), simulations_run=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
