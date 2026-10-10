"""端面几何锁参对照的只读汇总；没有完整终态时不产生精度结果。"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
R41 = HERE.parent
REFERENCE06 = R41 / '06_闸流水量一致性修复/G00_更严闸流一致性'
REUSED_SCRIPT = R41 / '07_修复结果汇总/汇总修复影响.py'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('_41_readonly_statistics', REUSED_SCRIPT)
stats07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats07)
require, read, sha, dump = stats07.require, stats07.read, stats07.sha, stats07.dump
CASES = ['reference', 'endpoint']
PROTOCOL_SHA256 = '4d5a980c0b6acff992f069b28749a0144c096b5bdd21ecb3b97f77ddea9cc041'
DOMAIN_PREFIXES = ['original_open_node', 'original_internal_link', 'endpoint_mean_vs_original_adjacent',
                   'endpoint_mean_vs_source', 'endpoint_internal_vs_source', 'endpoint_superjunction_vs_source']
DOMAIN_LABELS = ['原开渠内部节点水深与输入高度', '原内部连接平均水深与中点输入高度',
                 '端部有效平均水深与原相邻单元高度', '端部有效平均水深与源端面高度',
                 '端部内部节点水深与源端面高度', '端部主节点水深与源端面高度']
COUNT_ONLY_DOMAINS = {
    'endpoint_outside_either_contour_count': '两侧有效水深超旧新轮廓交集_09口径',
    'endpoint_raw_depth_outside_either_contour_count': '未乘theta的两侧原始水深超旧新轮廓交集',
    'endpoint_raw_depth_below_zero_count': '端面两侧原始水深出现负值',
}


def records(frame):
    # 仅把说明字段的缺省值序列化为null；模拟/观测有限性由读入核验单独保证。
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def run_status(folder):
    result_path = folder / '结果.json'
    progress_path = folder / '进度.json'
    record = read(result_path) if result_path.exists() else (read(progress_path) if progress_path.exists() else {'status': 'NOT_STARTED'})
    is_terminal = result_path.exists() and record['status'] in ['COMPLETED', 'FAILED']
    return dict(case=folder.name, terminal=is_terminal, status=record['status'],
                completed_cycles=record.get('completed_cycles', 0), error=record.get('error'),
                result_sha256=sha(result_path) if result_path.exists() else None)


def reference_reproduction():
    folder = HERE / 'reference'
    require(read(REFERENCE06 / '结果.json')['status'] == 'COMPLETED', '06参照本身未完整完成')
    checks = {}
    for name in ['半小时水位流量.csv', '实际位置水位.csv']:
        current = pd.read_csv(folder / name)
        old = pd.read_csv(REFERENCE06 / name)
        checks[name] = dict(rows=len(current), dataframe_exact_equal=current.equals(old),
                            current_sha256=sha(folder / name), reference_sha256=sha(REFERENCE06 / name))
        require(checks[name]['dataframe_exact_equal'], 'reference没有逐值复现06: ' + name)
    with np.load(folder / '末态.npz') as now, np.load(REFERENCE06 / '末态.npz') as old:
        states = {key: bool(np.array_equal(now[key], old[key])) for key in stats07.STATE_KEYS}
    require(all(states.values()), 'reference六类末态没有逐值复现06')
    return dict(water_level_tables=checks, final_six_states_exact=states,
                interpretation='只有reference复现06后，endpoint相对reference的差异才用于本轮归因。')


def identity_audit(configs, manifests):
    previous = read(REFERENCE06 / '运行配置.json')
    for case, cfg in configs.items():
        stats07.fingerprints(cfg)
        require(cfg['protocol']['flow_tol_m3s'] == 1e-9 and cfg['protocol']['head_tol_m'] == .0015,
                '两组必须使用同一flow_tol=1e-9及原head_tol')
        require(not cfg['assimilation'] and cfg['state_corrections'] == 0, '存在同化或状态校正')
        require(cfg['parameters_fixed'] and not cfg['source_geometry_changed'] and
                not cfg['internal_midpoint_geometry_changed'], '候选改动超出独立端面几何')
        require(cfg['endpoint_hook_applied'] == (case == 'endpoint'), '端面hook应用分支不符')
        for key in ['initial_state_sha256', 'parameter_sha256', 'forcing_array_sha256']:
            require(cfg[key] == previous[key], '初态/原参数/输入不等于06: ' + case + '/' + key)
        with np.load(HERE / case / '实际初始化_0s.npz') as z:
            require(all(stats07.arrsha(z[key]) == previous['initial_state_sha256'][key]
                        for key in stats07.STATE_KEYS), '实际保存初态不等于06')
    require(manifests['reference'] is None, 'reference不应应用端面修复hook')
    require(configs['reference']['internal_geometry_sha256'] == configs['endpoint']['internal_geometry_sha256'],
            '内部网格、床高或中点几何数组改变')
    manifest = manifests['endpoint']
    require(isinstance(manifest, dict) and manifest['revision'] == 'endpoint_source_geometry_41_v1',
            '端面hook身份不是09已审核候选')
    hook = R41 / '09_端部几何一致性核查/端部独立几何候选.py'
    require(manifest['hook_sha256'] == sha(hook), '端面hook源码指纹改变')
    require(manifest['closed_siphons_unchanged'] and not manifest['internal_midpoint_geometry_changed']
            and not manifest['dynamic_geometry_switch'] and manifest['original_parallel_geometry_called_first'],
            '候选越过声明的端面范围')
    require(manifest['fixed_endpoints'] == 2 * manifest['open_channels'] == len(manifest['endpoints']),
            '端面清单不完整')
    return dict(initial_six_states_exact_to_06=True, original_parameters_exact_to_06=True,
                all_five_Forcing_hashes_exact_to_06=True, common_flow_tol_m3s=1e-9,
                endpoint_hook_sha256=manifest['hook_sha256'], endpoint_manifest=manifest,
                internal_geometry_arrays_exact_equal=True,
                initial_geometric_volume_m3={c: configs[c]['initial_geometric_volume_m3'] for c in CASES},
                reference_has_no_endpoint_hook=True, assimilation=False)


def load_case_scores(case, observations):
    folder = HERE / case
    meters = pd.read_csv(folder / '测流位置算子流量.csv').rename(columns={'model_flow_m3s': 'sim'})
    stats07.check_grid(meters, ['station'], 44)
    meters['metric'] = 'flow_m3s'
    frame = stats07.load_scores(folder, case, observations, meters)
    expected = {'118闸节点水位': 118, '原88可比节点水位': 88, '43安装位置水位': 43,
                '旧更新34水位': 34, '旧9留出水位': 9, '44测流算子_描述性': 44}
    for name, part in stats07.groups(frame, include_additional_groups=True).items():
        require(len(part) == 1440 * expected[name], '分组覆盖不足: ' + name)
    return frame


def lanhe_water(case):
    folder = HERE / case
    index = read(folder / '逐成功子步文件索引.json')
    nodes = read(folder / index['nonboundary_node_metadata'])
    require(index['scalar_and_node_rows_aligned'] and nodes['count'] == len(nodes['ids']) == len(nodes['names']),
            '逐步控制体/节点索引未对齐')
    require(len(set(nodes['ids'])) == nodes['count'], '非边界主节点ID重复')
    files = [(item['file'], item['sha256']) for item in index['chunks']]
    instant, ledger, count, means = stats07.read_control_trace(files, case, nodes['count'])
    require(count == index['total_rows'] == sum(x['rows'] for x in index['chunks']), '逐步分块总行数不符')
    for row in ledger:
        row['geometric_storage_difference'] = '本兰河局部控制体未计算几何储水；全网另表，不能跨控制体相减。'
    saved = pd.read_csv(folder / '半小时测流至闸水量账.csv')
    require(np.array_equal(saved.time_s, stats07.TIMES) and (saved.duration_s == 1800).all(), '兰河水量账时轴不完整')
    require(np.max(abs(saved.model_meter_mean_m3s-means.model_mean_m3s)) < 1e-10,
            '兰河半小时均流不等于真实成功子步积分')
    residual = saved.model_meter_mean_m3s-saved.model_gate_mean_m3s-saved.offtake_mean_m3s-saved.native_storage_rate_m3s
    require(np.max(abs(residual-saved.continuity_residual_m3s)) < 1e-10, '半小时水量账算术不符')
    return instant, ledger, means, dict(successful_substeps=count, chunks=len(files),
                                       chunk_sha256={str(path): digest for path, digest in files})


def q16_temporal_scores(frames, observations, means_by_case):
    source = observations[(observations.station == 'JZZ16') & (observations.metric == 'flow_m3s')].sort_values('time_s')
    require(np.array_equal(source.time_s, np.arange(1441)*1800), 'Q16原始观测端点不足')
    observed_mean = (source.observed.to_numpy()[:-1]+source.observed.to_numpy()[1:])/2
    rows = []
    for frame in frames:
        case = str(frame.case.iloc[0])
        instant = frame[(frame.kind == 'meters') & (frame.station == 'JZZ16')].sort_values('time_s')
        require(len(instant) == 1440 and not instant.direct_forcing_observation.any(), 'Q16并非完整的非驱动测流检验')
        mean = means_by_case[case]
        require(np.array_equal(mean.time_s, stats07.TIMES), 'Q16均流时间轴不完整')
        for period, lo, hi in stats07.PERIODS:
            selected = (instant.time_s > lo) & (instant.time_s <= hi)
            average_selected = (mean.time_s > lo) & (mean.time_s <= hi)
            rows.append(dict(case=case, period=period, temporal_basis='半小时末态', unit='m3/s',
                             **stats07.stats(instant.loc[selected, 'error'])))
            rows.append(dict(case=case, period=period, temporal_basis='半小时时间均流', unit='m3/s',
                             **stats07.stats(mean.loc[average_selected, 'model_mean_m3s']-observed_mean[average_selected]),
                             note='模型用成功子步Q×dt积分；观测为相邻半小时样本的线性时间均值，不是新增高频实测。'))
    return rows


def assess_accuracy(group_changes, point_checks, protocol):
    settings = protocol['accuracy_assessment']
    tolerance = settings['numeric_comparison_tolerance']
    limits = settings['conditional_accuracy_gate']
    primary = {'118闸节点': '118闸节点水位', '43安装位置': '43安装位置水位', '旧9留出': '旧9留出水位'}
    require(set(settings['primary_water_groups']) == set(primary), '主精度组和运行前约定不同')
    require(limits['minimum_absolute_MAE_gain_m'] == .002 and limits['minimum_relative_MAE_gain_fraction'] == .05
            and limits['maximum_allowed_point_MAE_increase_m'] == .01, '实际精度门槛不是最终预定口径')
    checks = []
    for period, _, _ in stats07.PERIODS:
        per_group = []
        for group in primary.values():
            rows = [x for x in group_changes if x['period'] == period and x['score_group'] == group]
            require(len(rows) == 1, '主精度组缺失或重复')
            row = rows[0]
            gain = row['MAE_reference']-row['MAE_endpoint']
            threshold = max(limits['minimum_absolute_MAE_gain_m'],
                            limits['minimum_relative_MAE_gain_fraction']*row['MAE_reference'])
            per_group.append(dict(score_group=group, delta_MAE_m=row['delta_MAE'], delta_RMSE_m=row['delta_RMSE'],
                 MAE_RMSE_no_worse=row['delta_MAE'] <= tolerance and row['delta_RMSE'] <= tolerance,
                 MAE_gain_m=gain, required_gain_m=threshold, meaningful_gain=gain+tolerance >= threshold))
        point = point_checks[period]
        checks.append(dict(period=period, primary_groups=per_group,
             all_primary_MAE_RMSE_no_worse=all(x['MAE_RMSE_no_worse'] for x in per_group),
             at_least_one_meaningful_primary_gain=any(x['meaningful_gain'] for x in per_group),
             maximum_water_point_MAE_increase_m=point['maximum_increase_m'],
             all_water_point_MAE_increase_within_1cm=point['maximum_increase_m'] <= .01+tolerance))
    passed = all(x['all_primary_MAE_RMSE_no_worse'] and x['at_least_one_meaningful_primary_gain']
                 and x['all_water_point_MAE_increase_within_1cm'] for x in checks)
    return dict(passed=passed, numeric_comparison_tolerance=tolerance, period_checks=checks,
                point_scope='118节点与43安装位置逐点覆盖全部水位表示；88/34/9为子集，不重复计数。',
                qualification='仅本配对的条件性精度门槛，不等于全域物理验收、基线替换或部署。')


def write_accuracy(frames, protocol):
    overall, pointwise = stats07.score_tables(frames, include_additional_groups=True)
    paired = stats07.changes_table(pointwise, 'reference', 'endpoint')
    flow_supplement = []
    for frame in frames:
        flow = frame[frame.kind == 'meters']
        subsets = [('JZZ17端部最近中心近似测流', flow[flow.station == 'JZZ17'], 1),
                   ('JZZ59端部最近中心近似测流', flow[flow.station == 'JZZ59'], 1),
                   ('剔除2个已知端部近似后的42测流_描述性', flow[~flow.station.isin(['JZZ17', 'JZZ59'])], 42)]
        for period, lo, hi in stats07.PERIODS:
            for name, values, count in subsets:
                selected = values[(values.time_s > lo) & (values.time_s <= hi)]
                require(len(selected) == count * ((hi-lo)//1800), '流量补充分组覆盖不足')
                flow_supplement.append(dict(case=str(frame.case.iloc[0]), period=period, score_group=name,
                                            unit='m3/s', points=count, **stats07.stats(selected.error)))
    overall = pd.concat([overall, pd.DataFrame(flow_supplement)], ignore_index=True)
    overall.to_csv(HERE / '端面几何锁参_分组精度.csv', index=False)
    pointwise.to_csv(HERE / '端面几何锁参_逐测点精度.csv', index=False)
    paired.to_csv(HERE / '端面几何锁参_逐点变化.csv', index=False)
    paired[paired.water_MAE_worse_over_1cm].to_csv(HERE / '端面几何锁参_水位MAE退化超过1cm.csv', index=False)
    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata[metadata.kind == 'meters'][['case', 'station', 'chainage_m', 'source_name',
                  'direct_forcing_observation', 'end_connection_approximation', 'end_connection_offset_m']].drop_duplicates()
    metadata.to_csv(HERE / '44测流输入来源与端部近似.csv', index=False)
    keys = ['period', 'score_group', 'unit', 'points', 'samples']
    changes = overall[overall.case == 'endpoint'].merge(overall[overall.case == 'reference'],
                      on=keys, suffixes=('_endpoint', '_reference'), validate='one_to_one')
    for key in ['MAE', 'RMSE', 'bias', 'max_abs']:
        changes['delta_' + key] = changes[key + '_endpoint']-changes[key + '_reference']
    point_checks = {}
    for period, _, _ in stats07.PERIODS:
        use = paired[(paired.period == period) & paired.score_group.isin(['118闸节点水位', '43安装位置水位'])]
        require(len(use) == 161, '所有水位表示的逐点配对不完整')
        point_checks[period] = dict(maximum_increase_m=float(use.delta_MAE.max()),
                                    worst_point=records(use.loc[[use.delta_MAE.idxmax()]])[0])
    return dict(group_accuracy=records(overall), group_changes=records(changes), water_point_checks=point_checks,
                water_point_MAE_worse_over_1cm=records(paired[paired.water_MAE_worse_over_1cm]),
                conditional_accuracy_gate=assess_accuracy(records(changes), point_checks, protocol),
                flow_subgroup_notes={'44测流': '保留总组描述性统计，不称全部独立同位置现场验证。',
                    'JZZ17': '请求327550m，最近连接中心327583m，下游偏移33m，单列误差。',
                    'JZZ59': '请求1157902m，最近连接中心1157862m，上游偏移40m，单列误差。',
                    '其余42': '仅剔除上述已知端部近似；不据此确认全部现场仪器身份、断面或计量真值。'})


def check_domains(frame, definition):
    limits = {'original_open_node': len(definition['open_internal_node_ids']),
              'original_internal_link': len(definition['open_internal_link_ids'])}
    limits.update({key: len(definition['endpoints']) for key in DOMAIN_PREFIXES[2:]})
    columns = [key + suffix for key in DOMAIN_PREFIXES for suffix in ['_above_count', '_max_excess_m']]
    columns += list(COUNT_ONLY_DOMAINS)
    require(np.isfinite(frame[columns]).all().all(), '轮廓描述含非有限值')
    for key in DOMAIN_PREFIXES:
        count, excess = frame[key + '_above_count'], frame[key + '_max_excess_m']
        require(((count >= 0) & (count <= limits[key]) & (count == np.floor(count))).all(), '域外数量不符合固定节点集合')
        require(np.array_equal(count > 0, excess > definition['range_reporting_tolerance_m']), '域外数量与最大值不一致')
    for key in COUNT_ONLY_DOMAINS:
        count = frame[key]
        require(((count >= 0) & (count <= len(definition['endpoints'])) & (count == np.floor(count))).all(), '域外合计数量不符合固定端面集合')
    require((frame.endpoint_raw_depth_below_zero_count <= frame.endpoint_raw_depth_outside_either_contour_count).all(), '原始负水深未包含在原始域外合计中')


def domain_and_network_steps(case, cfg):
    folder = HERE / case
    definition = read(folder / '原始与端面轮廓定义.json')
    require(not definition['dynamic_geometry_switch'] and definition['range_is_not_engineering_crest']
            and definition['range_reporting_tolerance_m'] == 1e-9, '轮廓定义或用途被改变')
    sampled = pd.read_csv(folder / '半小时原始与端面轮廓范围.csv')
    require(len(sampled) == 1440 and np.array_equal(sampled.time_s, stats07.TIMES), '半小时轮廓记录不完整')
    check_domains(sampled, definition)
    initial = pd.DataFrame([read(folder / '初始轮廓范围.json')])
    check_domains(initial, definition)
    require(initial.endpoint_outside_either_contour_count.iloc[0] == 78, '0s的09有效水深域外口径不是已核验78端面')
    index = read(folder / '逐成功子步文件索引.json')
    accumulated = {(period, key): dict(samples=0, duration_s=0., samples_with_any_excess=0,
                   point_time_exceedances=0, duration_with_any_excess_s=0., maximum_excess_m=-np.inf, maximum_count=0)
                   for period, _, _ in stats07.PERIODS for key in DOMAIN_PREFIXES}
    count_totals = {(period, key): dict(samples=0, duration_s=0., samples_with_any_excess=0,
                    point_time_exceedances=0, duration_with_any_excess_s=0., maximum_count=0)
                    for period, _, _ in stats07.PERIODS for key in COUNT_ONLY_DOMAINS}
    network = {period: dict(net=0., storage=0., residual=0., maximum_rate=0.) for period, _, _ in stats07.PERIODS}
    integral_parts = []
    for chunk in index['chunks']:
        require(sha(chunk['file']) == chunk['sha256'], '逐步分块hash改变')
        with np.load(stats07.path_of(chunk['file'])) as z:
            columns = ['time_s', 'interval_start_s', 'dt_s', 'net_boundary_input_m3',
                       'full_network_discrete_storage_change_m3', 'full_network_discrete_residual_m3',
                       'full_network_discrete_residual_m3s']
            columns += [key + suffix for key in DOMAIN_PREFIXES for suffix in ['_above_count', '_max_excess_m']]
            columns += list(COUNT_ONLY_DOMAINS)
            a = pd.DataFrame({key: z[key] for key in columns})
        require(np.isfinite(a).all().all(), '逐步全网水量/轮廓记录非有限')
        check_domains(a, definition)
        residual = a.full_network_discrete_storage_change_m3-a.net_boundary_input_m3
        require(np.max(abs(residual-a.full_network_discrete_residual_m3)) < 1e-8 and
                np.max(abs(residual/a.dt_s-a.full_network_discrete_residual_m3s)) < 1e-8,
                '全网逐步水量账算术不符')
        integral = pd.DataFrame(dict(end_s=np.ceil(a.time_s/1800)*1800, duration_s=a.dt_s,
             net_boundary_input_m3=a.net_boundary_input_m3,
             discrete_continuity_storage_change_m3=a.full_network_discrete_storage_change_m3))
        integral_parts.append(integral.groupby('end_s').sum())
        for period, lo, hi in stats07.PERIODS:
            use = a[(a.interval_start_s >= lo) & (a.time_s <= hi)]
            if use.empty:
                continue
            n = network[period]
            n['net'] += float(use.net_boundary_input_m3.sum())
            n['storage'] += float(use.full_network_discrete_storage_change_m3.sum())
            n['residual'] += float(use.full_network_discrete_residual_m3.sum())
            n['maximum_rate'] = max(n['maximum_rate'], float(use.full_network_discrete_residual_m3s.abs().max()))
            for key in DOMAIN_PREFIXES:
                v = accumulated[(period, key)]
                has_excess = use[key + '_above_count'] > 0
                v['samples'] += len(use)
                v['duration_s'] += float(use.dt_s.sum())
                v['samples_with_any_excess'] += int(has_excess.sum())
                v['point_time_exceedances'] += int(use[key + '_above_count'].sum())
                v['duration_with_any_excess_s'] += float(use.loc[has_excess, 'dt_s'].sum())
                v['maximum_excess_m'] = max(v['maximum_excess_m'], float(use[key + '_max_excess_m'].max()))
                v['maximum_count'] = max(v['maximum_count'], int(use[key + '_above_count'].max()))
            for key in COUNT_ONLY_DOMAINS:
                v = count_totals[(period, key)]
                has_excess = use[key] > 0
                v['samples'] += len(use)
                v['duration_s'] += float(use.dt_s.sum())
                v['samples_with_any_excess'] += int(has_excess.sum())
                v['point_time_exceedances'] += int(use[key].sum())
                v['duration_with_any_excess_s'] += float(use.loc[has_excess, 'dt_s'].sum())
                v['maximum_count'] = max(v['maximum_count'], int(use[key].max()))
    integrals = pd.concat(integral_parts).groupby(level=0).sum().reset_index()
    saved = pd.read_csv(folder / '全网半小时水量账.csv')
    require(np.array_equal(integrals.end_s, stats07.TIMES) and (integrals.duration_s == 1800).all(), '全网半小时积分覆盖不足')
    for key in ['net_boundary_input_m3', 'discrete_continuity_storage_change_m3']:
        require(np.max(abs(saved[key]-integrals[key])) < 1e-6, '全网半小时账与真实成功步积分不符: ' + key)
    water = stats07.global_water(folder, case, {'candidate_initial_volume_m3': cfg['initial_geometric_volume_m3']})
    rows = []
    for period, lo, hi in stats07.PERIODS:
        for key, label in zip(DOMAIN_PREFIXES, DOMAIN_LABELS):
            values = accumulated[(period, key)]
            require(values['duration_s'] == hi-lo, '逐步轮廓记录时长不足')
            rows.append(dict(case=case, period=period, scope=label, sampling='全部成功子步', **values))
            selected = sampled[(sampled.time_s > lo) & (sampled.time_s <= hi)]
            rows.append(dict(case=case, period=period, scope=label, sampling='半小时末态抽样', samples=len(selected),
                 samples_with_any_excess=int((selected[key+'_above_count'] > 0).sum()),
                 point_time_exceedances=int(selected[key+'_above_count'].sum()),
                 maximum_count=int(selected[key+'_above_count'].max()),
                 maximum_excess_m=float(selected[key+'_max_excess_m'].max())))
        for key, label in COUNT_ONLY_DOMAINS.items():
            values = count_totals[(period, key)]
            require(values['duration_s'] == hi-lo, '逐步域外合计记录时长不足')
            rows.append(dict(case=case, period=period, scope=label, sampling='全部成功子步', **values))
            selected = sampled[(sampled.time_s > lo) & (sampled.time_s <= hi)]
            rows.append(dict(case=case, period=period, scope=label, sampling='半小时末态抽样', samples=len(selected),
                 samples_with_any_excess=int((selected[key] > 0).sum()),
                 point_time_exceedances=int(selected[key].sum()), maximum_count=int(selected[key].max())))
    return water, rows, dict(definition=definition, initial_range=initial.iloc[0].to_dict(),
                             step_network_totals=network, interpretation='数量为固定输入轮廓的域外点次；不当作不同设施数、数值失败或现场漫顶。')


def final_result_checks(case, control_rows, chunk_check):
    folder = HERE / case
    result = read(folder / '结果.json')
    require(result['status'] == 'COMPLETED' and result['full_30day_completed'] and result['completed_cycles'] == 1440
            and result['model_time_s'] == 2592000, '不是完整30天真实终态')
    require(result['parameters_unchanged'] and result['internal_geometry_unchanged'] and result['source_hashes_unchanged']
            and not result['assimilation'] and not result['baseline_replaced'], '终态完整性/无同化声明不符')
    require(result['completed_successful_substeps'] == chunk_check['successful_substeps'], '成功步总数不符')
    whole = next(x for x in control_rows if x['period'] == '全30天')
    require(result['all_successful_steps_control_volume_gate_pass'] == (whole['cv_steps_over_1e6'] == 0), '兰河控制体门槛标志不符')
    require(result['all_successful_steps_all_nonboundary_nodes_under_1e6'] ==
            (whole['max_nonboundary_node_residual_m3s'] <= 1e-6), '逐主节点门槛标志不符')
    failed = [json.loads(line) for line in (folder / '减步失败尝试.jsonl').read_text().splitlines() if line.strip()]
    require(len(failed) == result['rejected_attempts'] == result['retries'], '被拒绝尝试遗漏')
    require(all(x['six_state_rollback_exact'] and x['rolled_back_time_s'] == x['start_s'] for x in failed), '失败回退不完整')
    require(sum(result['iteration_counts'].values()) == chunk_check['successful_substeps'], '成功迭代次数分布总数不符')
    return dict(successful_steps=chunk_check['successful_substeps'], successful_iteration_counts=result['iteration_counts'],
                maximum_successful_iterations=whole['max_successful_iterations'], rejected_attempts=len(failed),
                rejected_iteration_counts=[x['iterations'] for x in failed], all_rejected_states_and_time_rolled_back=True,
                CV_gate_pass=result['all_successful_steps_control_volume_gate_pass'],
                nonboundary_node_gate_pass=result['all_successful_steps_all_nonboundary_nodes_under_1e6'])


def summarize(check_only):
    statuses = {case: run_status(HERE / case) for case in CASES}
    if not all(x['terminal'] for x in statuses.values()):
        return dict(status='WAITING_FOR_BOTH_TRUE_TERMINALS', cases=statuses, precision_results_written=False)
    if not all(x['status'] == 'COMPLETED' and x['completed_cycles'] == 1440 for x in statuses.values()):
        result = dict(status='INCOMPLETE_OR_FAILED_NO_FULL_MONTH_SCORE', cases=statuses, partial_results_scored=False,
                      precision_results_written=False, baseline_replaced=False, engineering_acceptance=False)
        if not check_only:
            dump(HERE / '端面几何锁参验证结论.json', result)
        return result
    configs = {case: read(HERE / case / '运行配置.json') for case in CASES}
    protocol_path = HERE / '运行前约定.json'
    require(sha(protocol_path) == PROTOCOL_SHA256, '最终运行前约定指纹改变')
    protocol = read(protocol_path)
    require(all(cfg['protocol'] == protocol for cfg in configs.values()), '运行配置没有使用最终精度门槛约定')
    identity = identity_audit(configs, {c: configs[c]['endpoint_geometry_manifest'] for c in CASES})
    exact = reference_reproduction()
    saved_exact = read(HERE / 'reference/与06逐值复现核验.json')
    require(saved_exact['all_exact'] and all(saved_exact['checks'].values()), 'reference复现自检没有通过')
    for name in ['实际位置观测算子.json', '测流位置算子与适用范围.json', '原始与端面轮廓定义.json']:
        require(read(HERE / 'reference' / name) == read(HERE / 'endpoint' / name), '两组算子或固定轮廓定义不同: ' + name)
    observations = stats07.observations()
    membership = stats07.numerical_group_membership(observations)
    frames, local_water, global_water, domains, means = [], [], [], [], {}
    case_checks, domain_definitions, sources = {}, {}, {}
    for case in CASES:
        frame = load_case_scores(case, observations)
        instant, ledger, means[case], chunks = lanhe_water(case)
        from_csv = frame[(frame.kind == 'meters') & (frame.station == 'JZZ16')].sort_values('time_s')
        require(np.max(abs(instant.sim.to_numpy()-from_csv.sim.to_numpy())) < 1e-10, '44点CSV中的Q16不等于逐步末态切口')
        water, domain, definition = domain_and_network_steps(case, configs[case])
        case_checks[case] = final_result_checks(case, ledger, chunks)
        case_checks[case]['chunks'] = chunks
        domain_definitions[case] = definition
        frames.append(frame); local_water += ledger; global_water += water; domains += domain
        sources.update(configs[case]['source_hashes'])
        sources.update({str(p): sha(p) for p in (HERE / case).iterdir() if p.suffix in ['.json', '.csv', '.npz', '.jsonl']})
    temporal = q16_temporal_scores(frames, observations, means)
    sources.update(membership['source_sha256'])
    sources[str(REUSED_SCRIPT)] = sha(REUSED_SCRIPT)
    sources[str(Path(__file__))] = sha(Path(__file__))
    result = dict(status='COMPLETE_LOCKED_PARAMETER_PAIRED_DESCRIPTION', cases=statuses, identity=identity,
                  reference_exact_to_06=exact, score_group_membership=membership, runtime_checks=case_checks,
                  Q16_temporal_scores=temporal, lanhe_water=local_water, whole_network_water=global_water,
                  contour_definitions_and_initial_ranges=domain_definitions,
                  source_sha256=sources, precision_results_written=not check_only,
                  baseline_replaced=False, engineering_acceptance=False,
                  limitations=['两组均原n/C、相同初态与历史输入、flow_tol=1e-9，无同化；本轮只比较09端面hook。',
                               '44测流描述性列出，输入来源重合标志及JZZ17/59端部近似单列；不能统称44个严格同位置独立盲测。',
                               'Q16末态误差与半小时积分均流分开；相邻实测端点的时间均值不是新增高频实测。',
                               '全网和兰河控制体不同，残差符号不同；几何-离散储水差不是现场漏水或补水。',
                               '几何域外数量为固定输入轮廓描述，不是现场堤顶或漫顶判定；不通过抬高轮廓/按水深切换几何取得完成。'])
    if not check_only:
        result.update(write_accuracy(frames, protocol))
        pd.DataFrame(temporal).to_csv(HERE / 'Q16末态与半小时均流分别计分.csv', index=False)
        pd.DataFrame(local_water).to_csv(HERE / '兰河离散水量账.csv', index=False)
        pd.DataFrame(global_water).to_csv(HERE / '全网离散与几何储水账.csv', index=False)
        pd.DataFrame(domains).to_csv(HERE / '原始与端面轮廓域外描述.csv', index=False)
        dump(HERE / '端面几何锁参验证结论.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='只读终态与接口核验，不写任何精度结果')
    args = parser.parse_args()
    try:
        result = summarize(args.check_only)
    except Exception as exc:
        print(json.dumps(dict(status='BLOCKED_BY_ARTIFACT_OR_IDENTITY_CHECK', error=repr(exc),
                              precision_results_written=False), ensure_ascii=False))
        return 2
    print(json.dumps(dict(status=result['status'], cases=result['cases'],
                         precision_results_written=result['precision_results_written']), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
