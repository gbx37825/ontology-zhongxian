"""只读40配对运行。35分支全部终态后汇总；--check-only审计已有完成分支但不写文件。"""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
R40 = HERE.parent
R = R40.parent
WORKSPACE = R.parents[1]
B = R / '26_冻结基线与后续归因'
METHODS = ['F', 'P', 'W', 'S', 'H']
TRACKS = ['known_historical_inputs', 'frozen_at_issue']
LEADS = [1800, 3600, 10800, 21600]
OP_KEYS = ['scope', 'group', 'station', 'metric']
GROUP_SIZES = {'旧留出9': 9, '更新34': 34, '安装位置43': 43,
               '闸节点118': 118, '原可比节点88': 88, '兰河测流Q16': 1}


def path_of(value):
    p = Path(value)
    return p if p.is_absolute() else WORKSPACE / p


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(None)
def sha(path):
    return hashlib.sha256(path_of(path).read_bytes()).hexdigest()


def arrsha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def classify(result):
    status = result.get('status')
    if status == 'COMPLETED':
        return 'COMPLETED'
    if status != 'FAILED':
        return 'IN_PROGRESS'
    text = result.get('error', '') + '\n' + result.get('traceback', '')
    numerical = re.search(r'Hydraulic step did not converge|FloatingPointError|非有限|水深低于下限', text)
    return 'FAILED_NUMERICAL' if numerical else 'FAILED_OTHER_REQUIRES_ERROR_REVIEW'


def inventory(cuts):
    rows = []
    for issue in cuts:
        for method in METHODS:
            folder = R40 / '06_配对运行' / f'起点_{issue}s' / method
            result_path = folder / '结果.json'
            record = read(result_path) if result_path.exists() else (
                read(folder / '进度.json') if (folder / '进度.json').exists() else {})
            terminal = result_path.exists() and record.get('status') in ['COMPLETED', 'FAILED']
            rows.append(dict(issue_s=issue, method=method, folder=str(folder),
                             status=classify(record) if record else 'NOT_STARTED', terminal=terminal,
                             completed_adaptation_cycles=record.get('completed_adaptation_cycles'),
                             phase=record.get('phase'), error=record.get('error'),
                             traceback=record.get('traceback'), result_sha256=sha(result_path) if result_path.exists() else None))
    archived = []
    archive = R40 / '05_先行核验/首轮坐标一致性停止'
    explanation = read(archive / '归档说明.json') if (archive / '归档说明.json').exists() else None
    for p in sorted(archive.glob('起点_*s/*/结果.json')):
        r = read(p)
        archived.append(dict(path=str(p), sha256=sha(p), status='ARCHIVED_SETUP_FAILURE',
                             original_status=r['status'], method=r['method'], issue_s=r['issue_s'],
                             completed_adaptation_cycles=r['completed_adaptation_cycles'], error=r.get('error'),
                             explanation=explanation, included_in_35_case_denominator=False))
    return rows, archived


def group_frames(frame):
    mapped = frame.scope == '安装位置43'
    nodes = frame.scope == '闸节点118'
    return {'旧留出9': frame[mapped & (frame.group == '留出9')],
            '更新34': frame[mapped & (frame.group == '更新34')], '安装位置43': frame[mapped],
            '闸节点118': frame[nodes], '原可比节点88': frame[nodes & frame.pdf_comparable],
            '兰河测流Q16': frame[frame.scope == '兰河测流切口']}


def audit_case(folder, observation, entry, protocol, reference_config):
    cfg = read(folder / '运行配置.json')
    result = read(folder / '结果.json')
    issue, method = int(cfg['issue_s']), cfg['method']
    start = issue - 21600
    times = np.arange(start + 1800, issue + 1, 1800)
    require(result['status'] == 'COMPLETED' and result['completed_adaptation_cycles'] == 12, '完成标记/12适应周期不符')
    require(cfg['adaptation_start_s'] == start == entry['adaptation_start_s'], '适应开始时刻不符')
    require(cfg['protocol'] == protocol and cfg['members'] == 50 and cfg['seed'] == 20260923, '协议/成员/seed不符')
    require(cfg['source_hashes'] == reference_config['source_hashes'], '配对源码或输入身份不同')
    require(all(sha(p) == h for p, h in cfg['source_hashes'].items()), '当前来源hash已改变')
    require(cfg['score_operators'] == reference_config['score_operators'], '配对评分算子不同')
    require(cfg['update_operators'] == reference_config['update_operators'], '配对更新算子不同')
    require(cfg['common_state_sha256'] == entry['sha256'] == sha(cfg['common_state_path']), '共同快照hash不符')
    require(cfg['initial_states_sha256'] == entry['state_sha256'], '初始水力状态身份不符')
    with np.load(path_of(cfg['common_state_path'])) as z:
        require(float(z['time_s']) == start, '共同快照时刻不符')
        require(all(arrsha(z[k]) == h for k, h in entry['state_sha256'].items()), '共同快照六状态逐数组hash不符')
    with np.load(folder / '初始参数集合.npz') as z:
        initial = z['theta'].copy()
    require(initial.shape == (119, 50) and arrsha(initial) == cfg['initial_theta_sha256'], '初始theta身份不符')
    require(cfg['initial_theta_sha256'] == reference_config['initial_theta_sha256'], '配对初始theta不同')
    with np.load(folder / '参数集合轨迹.npz') as z:
        theta, theta_times = z['theta'].copy(), z['time_s'].copy()
    require(theta.shape == (13, 119, 50) and np.array_equal(theta_times, np.r_[start, times]), '参数集合轨迹形状/时间不符')
    require(np.isfinite(theta).all() and np.array_equal(theta[0], initial), '参数轨迹初值或有限性不符')
    with np.load(folder / '发行时集合检查点.npz') as z:
        require(float(z['time_s']) == issue and np.array_equal(z['theta'], theta[-1]), '发行检查点theta/时刻不符')
    require(all(result[k] == 0 for k in ['future_observations_used_by_algorithm', 'future_parameter_updates', 'future_state_updates']), '发行后出现更新计数')
    require(result['forecast_tracks'] == TRACKS and result['forecast_leads_s'] == LEADS, '预测轨道/提前量不符')

    ops = pd.DataFrame(cfg['score_operators'])[OP_KEYS]
    require(len(ops) == 162 and not ops.duplicated().any(), '评分算子不是162个唯一标识')
    future = pd.read_csv(folder / '冻结更新预测评分.csv')
    adaptation = pd.read_csv(folder / '适应期先验与分析.csv')
    require(len(future) == result['forecast_rows'] == 1296 and len(adaptation) == 3888, '预测/适应期行数不符')
    for frame in [future, adaptation]:
        require((frame.issue_s == issue).all(), '评分发行时刻不符')
        require(np.isfinite(frame[['sim_mean', 'ensemble_std', 'observed']]).all().all(), '评分含非有限值')
        keys = pd.MultiIndex.from_arrays([frame.station, frame.metric, frame.time_s])
        observed = observation.reindex(keys)
        require(observed.valid.notna().all() and observed.valid.all(), '目标观测缺失或无效')
        require(np.max(abs(frame.observed.to_numpy() - observed.value.to_numpy())) < 1e-10, 'CSV观测不等于冻结观测')
        require(np.array_equal(frame.pdf_comparable, observed.pdf_comparable.to_numpy()), '88点原可比标记不符')
    require((future.phase == 'no_updates_after_issue').all(), '未来评分混入分析或先验')
    require(not future.duplicated(['track', 'lead_s', *OP_KEYS]).any(), '预测行重复')
    require(not adaptation.duplicated(['time_s', 'phase', *OP_KEYS]).any(), '适应期行重复')
    require(set(adaptation.phase) == {'forecast_before_update', 'analysis_after_update'}, '适应期phase不符')
    require((adaptation.lead_s == 0).all(), '适应期误标未来提前量')
    prediction_errors = []
    with np.load(folder / '各成员未来预测.npz') as z:
        predictions = z['predictions']
        require(predictions.shape == (2, 4, 50, 162) and np.isfinite(predictions).all(), '成员预测数组形状/有限性不符')
        require(np.array_equal(z['lead_s'], LEADS), '成员预测提前量不符')
        for it, track in enumerate(TRACKS):
            for j, lead in enumerate(LEADS):
                block = future[(future.track == track) & (future.lead_s == lead)].reset_index(drop=True)
                require(block[OP_KEYS].equals(ops) and (block.time_s == issue + lead).all(), '预测CSV算子顺序或valid时刻不符')
                mean_error = float(abs(predictions[it, j].mean(axis=0) - block.sim_mean).max())
                std_error = float(abs(predictions[it, j].std(axis=0, ddof=1) - block.ensemble_std).max())
                require(max(mean_error, std_error) < 1e-10, '成员NPZ均值/标准差与CSV不符')
                require(all(len(part) == GROUP_SIZES[name] for name, part in group_frames(block).items()), '9/34/43/118/88/Q16分组数不符')
                prediction_errors.append(dict(track=track, lead_s=lead, mean_max_abs_difference=mean_error, std_max_abs_difference=std_error))
    for now in times:
        blocks = [adaptation[(adaptation.time_s == now) & (adaptation.phase == phase)].reset_index(drop=True)
                  for phase in ['forecast_before_update', 'analysis_after_update']]
        require(all(block[OP_KEYS].equals(ops) for block in blocks), '适应期每时刻162算子不完整')
        if method in ['F', 'P', 'W']:
            require(np.array_equal(blocks[0][['sim_mean', 'ensemble_std']], blocks[1][['sim_mean', 'ensemble_std']]), '仅参数路线即时读数被更改')
    if method in ['F', 'S']:
        require(np.array_equal(theta, np.broadcast_to(initial, theta.shape)), 'F/S参数集合发生改变')

    prior = cfg['prior']
    center = np.array(list(prior['centers'].values()))
    lower = np.maximum(np.asarray(prior['physical_lower_ratios']), center * .95)
    upper = np.minimum(np.asarray(prior['physical_upper_ratios']), center * 1.05)
    require(np.allclose(lower, prior['effective_lower_ratios'], atol=1e-14, rtol=0) and
            np.allclose(upper, prior['effective_upper_ratios'], atol=1e-14, rtol=0), '有效域并非原物理界与中心±5%交集')
    ratio = np.exp(theta)
    require((ratio >= lower[None, :, None] - 1e-12).all() and (ratio <= upper[None, :, None] + 1e-12).all(), '参数越过有效域')
    parameter_csv = pd.read_csv(folder / '参数轨迹.csv')
    names = list(prior['centers'])
    require(len(parameter_csv) == 13 * 119, '参数统计CSV行数不符')
    for i, now in enumerate(theta_times):
        block = parameter_csv[parameter_csv.time_s == now].reset_index(drop=True)
        require(block.parameter.tolist() == names, '参数统计顺序不符')
        for key, expected in [('mean_ratio', ratio[i].mean(axis=1)), ('minimum_ratio', ratio[i].min(axis=1)),
                              ('maximum_ratio', ratio[i].max(axis=1)), ('std_log', theta[i].std(axis=1, ddof=1))]:
            require(np.max(abs(block[key] - expected)) < 1e-10, '参数CSV与成员数组不符: ' + key)
        require(np.allclose(block.effective_lower, lower, rtol=0, atol=1e-12) and
                np.allclose(block.effective_upper, upper, rtol=0, atol=1e-12), '参数CSV有效域不符')
        require(np.array_equal(block.at_lower, (theta[i] <= np.log(lower)[:, None] + 1e-12).sum(axis=1)) and
                np.array_equal(block.at_upper, (theta[i] >= np.log(upper)[:, None] - 1e-12).sum(axis=1)), '参数CSV触界数不符')
    parameter_csv['issue_s'], parameter_csv['method'] = issue, method

    ledger = read(folder / '更新水量与参数审计.json')
    windows = read(folder / '窗口记录.json')
    trace = read(folder / '逐观测更新记录.json')
    require(len(ledger) == 12 and [x['time_s'] for x in ledger] == times.tolist(), '12条更新账时间不符')
    expected_updates = {(o['station'], o['metric']) for o in cfg['update_operators']}
    require(len(expected_updates) == 34, '更新点身份不是34个唯一点')
    used, rebuilt_trace, accounts, batches = set(), [], [], []
    volume = np.zeros(50)
    absolute_step_mean_volume = 0.
    for i, row in enumerate(ledger):
        now, source_times = row['time_s'], row['source_observation_times_s']
        current_sha = arrsha(theta[i])
        rejuvenation, audit = row['rejuvenation'], row['analysis']
        for kind, item in [('rejuvenation', rejuvenation), ('analysis', None if audit is None else audit['parameter_analysis'])]:
            if item is None:
                continue
            require(item['before_theta_sha256'] == current_sha, 'analysis/rejuvenation参数哈希链断裂')
            current_sha = item['after_theta_sha256']
            if kind == 'rejuvenation':
                require(method in ['P', 'W', 'H'] and item['states_unchanged'] and item['clocks_unchanged'], '补散度改变状态/时钟或出现在错误方法')
            elif audit['update_parameters']:
                cap = audit['elapsed_cycles'] * np.log(1.01)
                require(abs(item['analysis_log_step_limit'] - cap) < 1e-14 and item['max_abs_log_change'] <= cap + 1e-12, 'analysis参数步幅超约定')
            accounts.append(dict(issue_s=issue, method=method, time_s=now, account=kind,
                                 max_abs_log_change=item['max_abs_log_change'], max_abs_relative_change=item['max_abs_relative_change'],
                                 raw_max_abs_log_change=item['raw_max_abs_log_change'],
                                 analysis_log_step_limit=item['analysis_log_step_limit'], step_limited_entries=item['step_limited_entries'],
                                 clipped_to_domain_entries=item['clipped_to_domain_entries'], at_lower_entries=item['at_lower_entries'],
                                 at_upper_entries=item['at_upper_entries'], rejuvenated_parameters=item.get('rejuvenated_parameters')))
        require(current_sha == arrsha(theta[i + 1]), '参数哈希链终值不等于保存轨迹')
        if audit is None:
            require(source_times == [], '无analysis却记录消费样本')
            continue
        require(method != 'F' and audit['observations'] == 34 and audit['clocks_unchanged'], '批次观测数/时钟不符')
        require(audit['single_prior_single_observation_batch'], '不是单一先验的联合批次')
        require(audit['update_parameters'] == (method in ['P', 'W', 'H']) and
                audit['update_states'] == (method in ['S', 'H']), '参数/状态更新通道不符')
        require(source_times and max(source_times) <= now <= issue and len(source_times) == len(set(source_times)), '重复或未来source_times')
        require(all(t in times for t in source_times), 'source_times越过本6h适应范围')
        require(audit['elapsed_cycles'] == len(source_times), '更新预算与原始周期数不符')
        require({(x['station'], x['metric']) for x in audit['trace']} == expected_updates and len(audit['trace']) == 34, '批次混入留出或缺失更新点')
        keys = {(station, metric, t) for station, metric in expected_updates for t in source_times}
        require(not keys & used, '原始观测样本重复消费')
        used.update(keys)
        accepted = sum(x['status'] == 'accepted' for x in audit['trace'])
        require(accepted == audit['accepted_observations'] and 34 - accepted == audit['rejected_observations'], '接受/拒收计数不符')
        for item in audit['trace']:
            require(item['observation_time_s'] == now and item['sigma'] == .03, '批次观测时刻或sigma不符')
            actual_y = np.mean([observation.loc[(item['station'], item['metric'], t), 'value'] for t in source_times])
            require(abs(actual_y - item['observed']) < 1e-10, '批次观测不等于对应原始样本均值')
            innovation = item['observed'] - item['forecast']
            limit = 6 * np.sqrt(item['forecast_spread']**2 + .03**2)
            require(abs(innovation - item['innovation']) < 1e-10 and abs(limit - item['innovation_limit']) < 1e-10, '创新或拒收界算术不符')
            require((item['status'] == 'accepted') == (item['observed'] > 0 and abs(innovation) <= limit), '6sigma拒收结果不符')
            require(item['used_for_parameters'] == (item['status'] == 'accepted' and audit['update_parameters']) and
                    item['used_for_states'] == (item['status'] == 'accepted' and audit['update_states']), '拒收观测仍被标为更新')
            rebuilt_trace.append({**item, 'batch_end_s': now, 'source_times_s': source_times, 'method': method})
        state = audit['state_analysis']
        require(state['masked_rows_unchanged'] and state['max_head_change_m'] <= .02 + 1e-12 and
                state['max_flow_change_m3s'] <= 3 + 1e-12, '状态mask/整批幅度不符')
        dv = np.asarray(state['delta_volume_members_m3'])
        require(dv.shape == (50,) and np.isfinite(dv).all(), '等效水量成员数组无效')
        require(abs(float(dv.mean()) - state['delta_volume_mean_m3']) < 1e-8, '等效水量均值不符')
        if method in ['P', 'W']:
            require(state['prior_state_sha256'] == state['posterior_state_sha256'] and not state['dirty'] and
                    np.array_equal(dv, np.zeros(50)), 'P/W即时水力状态或水量被改')
        if state['dirty']:
            require(np.allclose(np.asarray(state['geometric_volume_after_m3']) - state['geometric_volume_before_m3'], dv, rtol=0, atol=1e-8), '几何体积前后差与等效水量不符')
        volume += dv
        absolute_step_mean_volume += abs(float(dv.mean()))
        batches.append(dict(issue_s=issue, method=method, batch_end_s=now, source_times_s=json.dumps(source_times),
                            raw_samples_consumed=len(keys), accepted_batch_observations=accepted,
                            rejected_batch_observations=34 - accepted, elapsed_cycles=audit['elapsed_cycles'],
                            single_prior_single_observation_batch=True, Y_source=audit['Y_source']))
    require(trace == rebuilt_trace, '逐观测文件不等于更新账中的真实批次记录')
    expected_samples = 0 if method == 'F' else 408
    require(len(used) == result['consumed_update_samples'] == expected_samples, '原始样本消费数不符')
    if method != 'W':
        require(not windows, '非W方法存在窗口记录')
        require(len(batches) == (0 if method == 'F' else 12), '非W批次数不符')
    else:
        cursor = start
        for window in windows:
            require(window['window_start_s'] == cursor and window['classification_uses_only_input_times_at_or_before'] <= cursor, 'W窗口不连续或用开始后的信息选窗')
            features = window['features']
            require([x['span_s'] for x in features] == [21600, 10800, 3600], 'W回看范围被改变')
            for feature in features:
                if feature['complete']:
                    require(feature['end_s'] <= cursor and feature['start_s'] == feature['end_s'] - feature['span_s'], 'W输入特征含未来')
            chosen = next((x for x in features[:2] if x['complete'] and x['input_quiet']), features[-1])
            planned = int(chosen['span_s'] / 1800)
            actual = min(planned, int((issue - cursor) / 1800))
            require(window['planned_cycles'] == planned and window['actual_cycles'] == actual and
                    window['window_end_s'] == cursor + actual * 1800, 'W窗口长度与过去输入规则不符')
            batch = next(x for x in ledger if x['time_s'] == window['window_end_s'])
            require(batch['source_observation_times_s'] == list(np.arange(cursor + 1800, window['window_end_s'] + 1, 1800)), 'W样本不完整或跨窗')
            require(batch['analysis']['Y_source'] == 'provided_window_forecast', 'W历史观测未匹配同期forecast均值')
            cursor = window['window_end_s']
        require(cursor == issue and len(batches) == len(windows), 'W窗口未完整覆盖适应期')

    geometry = []
    for filename, dimensions, expected_rows in [
            ('预测数值与几何审计.csv', ['track', 'lead_s'], 400),
            ('适应期输入轮廓检查.csv', ['phase'], 1200)]:
        frame = pd.read_csv(folder / filename)
        require(len(frame) == expected_rows, '几何/数值审计行数不符: ' + filename)
        key_columns = ['track', 'lead_s', 'member'] if 'track' in frame else ['time_s', 'phase', 'member']
        require(not frame.duplicated(key_columns).any(), '几何/数值成员记录重复')
        require(set(frame.member) == set(range(50)), '几何/数值成员ID不完整')
        require(np.isfinite(frame.select_dtypes(include='number')).all().all(), '几何/数值记录非有限')
        if 'track' in frame:
            require(set(frame.track) == set(TRACKS) and set(frame.lead_s) == set(LEADS), '预测几何轨道/提前量不完整')
            require((frame.groupby(['track', 'lead_s']).size() == 50).all(), '预测几何每提前量不是50成员')
        else:
            require(set(frame.time_s) == set(times) and set(frame.phase) == {'forecast_before_update', 'analysis_after_update'}, '适应期几何时刻/阶段不完整')
            require((frame.groupby(['time_s', 'phase']).size() == 50).all(), '适应期几何每时刻不是50成员')
        for keys, block in frame.groupby(dimensions, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            geometry.append(dict(issue_s=issue, method=method, source=filename, **dict(zip(dimensions, keys)),
                                 sampled_member_states=len(block), member_states_above_input_contour=int((block.open_points_above_input_contour > 0).sum()),
                                 maximum_points_above_input_contour=int(block.open_points_above_input_contour.max()),
                                 maximum_excess_m=float(block.maximum_excess_m.max()), minimum_internal_depth_m=float(block.minimum_internal_depth_m.min()),
                                 retries=int(block.retries.sum()) if 'retries' in block else None,
                                 max_head_residual_m=float(block.max_head_residual_m.max()) if 'max_head_residual_m' in block else None,
                                 max_gate_residual_m3s=float(block.max_gate_residual_m3s.max()) if 'max_gate_residual_m3s' in block else None,
                                 interpretation='仅描述采样状态的既有输入轮廓超出；不作工程适用性通过判据'))
    water = dict(issue_s=issue, method=method, adaptation_start_s=start, duration_s=21600,
                 cumulative_delta_volume_mean_m3=float(volume.mean()), cumulative_delta_volume_min_member_m3=float(volume.min()),
                 cumulative_delta_volume_max_member_m3=float(volume.max()), sum_abs_batch_mean_delta_volume_m3=absolute_step_mean_volume,
                 cumulative_delta_volume_members_m3=volume.tolist(), interpretation='仅本origin独立6h适应期的状态更新等效蓄量；不得跨origin相加称整月补水')
    audit = dict(issue_s=issue, method=method, status='PASSED', completed_adaptation_cycles=12,
                 consumed_raw_update_samples=len(used), raw_consumption_includes_rejected_samples=True,
                 analysis_batches=len(batches), future_rows=1296, adaptation_rows=3888,
                 prediction_npz_csv_checks=prediction_errors, same_source_and_initial_identity=True,
                 instantaneous_state_unchanged=method in ['F', 'P', 'W'], parameter_collection_fixed=method in ['F', 'S'],
                 effective_domain_verified=True, W_nonoverlap_and_causal_choice_verified=method == 'W',
                 H_single_joint_batch_verified=method == 'H', initial_domain_clipped_entries=prior['clipped_entries'],
                 forecast_theta_evidence='发行检查点theta与末次适应theta逐值相同；结果的三项未来更新计数为0。未来NPZ只含预测值，不含逐提前量theta；无更新依赖冻结runner/worker逐成员参数断言，未宣称独立重算未来theta。',
                 source_hashes=cfg['source_hashes'], artifact_sha256={p.name: sha(p) for p in folder.iterdir() if p.suffix in ['.json', '.csv', '.npz']})
    future['method'], adaptation['method'] = method, method
    return dict(audit=audit, future=future, adaptation=adaptation, accounts=accounts, batches=batches,
                water=water, geometry=geometry, parameters=parameter_csv)


def metrics(frame, common_issues):
    rows = []
    populations = [('逐起点', int(issue), part) for issue, part in frame.groupby('issue_s')]
    if common_issues:
        populations.append(('全部共同完成起点', -1, frame[frame.issue_s.isin(common_issues)]))
    for population, issue, sample in populations:
        for keys, block in sample.groupby(['method', 'track', 'phase', 'lead_s'], dropna=False):
            method, track, phase, lead = keys
            for group, part in group_frames(block).items():
                if part.empty:
                    continue
                err = part.sim_mean.to_numpy() - part.observed.to_numpy()
                rows.append(dict(population=population, issue_s=issue, origin_count=part.issue_s.nunique(),
                                 method=method, track='' if pd.isna(track) else track, phase=phase, lead_s=int(lead),
                                 score_group=group, unit='m3/s' if group == '兰河测流Q16' else 'm', samples=len(err),
                                 MAE=float(abs(err).mean()), RMSE=float(np.sqrt(np.mean(err**2))), bias=float(err.mean()), max_abs=float(abs(err).max())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='只审计已有完成分支并打印状态；不写正式或临时汇总文件')
    args = parser.parse_args()
    protocol = read(R40 / '预定在线对照约定.json')
    selection = read(R40 / '04_起点与公共初态/预定起点.json')
    common = read(R40 / '04_起点与公共初态/公共初态清单.json')
    cuts = list(map(int, selection['cut_times_s']))
    require(cuts == protocol['issue_times_s'] and len(cuts) == 7, '不是预定7起点')
    require(list(protocol['methods']) == METHODS, '五方法身份改变')
    entries = {int(x['issue_s']): x for x in common['states']}
    require(sorted(entries) == cuts and common['status'] == 'COMPLETED', '公共初态清单不完整')
    cases, archived = inventory(cuts)
    completed = [x for x in cases if x['status'] == 'COMPLETED']
    observation = pd.read_csv(B / '共享输入/作者已核实修正观测层.csv').set_index(['station_code', 'metric', 'time_s'])
    require(not observation.index.duplicated().any(), '冻结观测主键重复')
    reference_config = read(Path(completed[0]['folder']) / '运行配置.json') if completed else None
    audits, data = [], []
    for case in completed:
        try:
            item = audit_case(Path(case['folder']), observation, entries[case['issue_s']], protocol, reference_config)
            audits.append(item['audit'])
            data.append(item)
        except Exception as exc:
            audits.append(dict(issue_s=case['issue_s'], method=case['method'], status='FAILED_ARTIFACT_AUDIT', error=repr(exc)))
    bad = [x for x in audits if x['status'] != 'PASSED']
    all_terminal = all(x['terminal'] for x in cases)
    common_issues = [issue for issue in cuts if all(any(x['issue_s'] == issue and x['method'] == method and x['status'] == 'PASSED' for x in audits) for method in METHODS)]
    status = dict(status='BLOCKED_ARTIFACT_AUDIT' if bad else ('READY_ALL_TERMINAL' if all_terminal else 'INCOMPLETE'),
                  expected_case_count=35, terminal_count=sum(x['terminal'] for x in cases),
                  completed_count=len(completed), failed_terminal_count=sum(x['terminal'] and x['status'] != 'COMPLETED' for x in cases),
                  audited_completed_count=len(data), common_completed_issue_times_s=common_issues,
                  cases=cases, archived_setup_failures=archived, artifact_audit_failures=bad,
                  formal_summary_generated_this_run=False, formal_metrics_written_this_run=False,
                  missing_or_failed_cases_not_dropped=True)
    if args.check_only:
        print(json.dumps({k: status[k] for k in ['status', 'expected_case_count', 'terminal_count', 'completed_count', 'audited_completed_count', 'artifact_audit_failures', 'common_completed_issue_times_s']}, ensure_ascii=False, indent=2))
        return 2 if bad else 0
    dump(HERE / '汇总状态.json', status)
    dump(HERE / '逐分支文件与机制审计.json', audits)
    if not all_terminal or bad:
        print(json.dumps(dict(status=status['status'], terminal_count=status['terminal_count'], expected=35, formal_metrics_written=False), ensure_ascii=False))
        return 2 if bad else 0
    if data:
        future = pd.concat([x['future'] for x in data], ignore_index=True)
        adaptation = pd.concat([x['adaptation'] for x in data], ignore_index=True)
        forecast_metrics = metrics(future, common_issues)
        adaptation_metrics = metrics(adaptation, common_issues)
        forecast_metrics.to_csv(HERE / '冻结更新预测分组误差.csv', index=False)
        adaptation_metrics.to_csv(HERE / '适应期先验与分析分组误差.csv', index=False)
        join = ['population', 'issue_s', 'origin_count', 'track', 'phase', 'lead_s', 'score_group', 'unit', 'samples']
        baseline = forecast_metrics[forecast_metrics.method == 'F'].drop(columns='method')
        paired = forecast_metrics[forecast_metrics.method != 'F'].merge(baseline, on=join, suffixes=('', '_F'), validate='many_to_one')
        for key in ['MAE', 'RMSE', 'bias', 'max_abs']:
            paired['delta_' + key + '_versus_F'] = paired[key] - paired[key + '_F']
        paired.to_csv(HERE / '相对F配对误差变化.csv', index=False)
        for filename, key in [('参数analysis与补散度分账.csv', 'accounts'), ('窗口与原始观测消费审计.csv', 'batches'), ('数值与输入轮廓描述.csv', 'geometry')]:
            rows = [row for item in data for row in item[key]]
            table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['issue_s', 'method', 'time_s', 'account'] if key == 'accounts' else ['issue_s', 'method', 'batch_end_s', 'raw_samples_consumed'])
            table.to_csv(HERE / filename, index=False)
        dump(HERE / '各起点独立六小时同化等效水量.json', [x['water'] for x in data])
        pd.concat([x['parameters'] for x in data], ignore_index=True).to_csv(HERE / '参数有效域与触界统计.csv', index=False)
        primary = forecast_metrics[(forecast_metrics.population == '全部共同完成起点') &
                                   (forecast_metrics.track == 'frozen_at_issue') & (forecast_metrics.score_group == '旧留出9')]
        require((primary.samples <= 63).all(), '主表每提前量超过7×9个点')
    else:
        primary = pd.DataFrame()
    status.update(status='COMPLETED_WITH_FAILED_CASES' if status['failed_terminal_count'] else 'COMPLETED',
                  formal_summary_generated_this_run=True, formal_metrics_written_this_run=bool(data))
    summary = dict(**status, primary_frozen_input_old9_by_lead=primary.to_dict('records'),
                   limitations=['35个分支全部列明，失败仍占原分母；总比较仅聚合五方法共同完成且审计通过的同一起点，并明确清单。',
                                '两输入轨道和9/34/43/118/88/Q16各组分开；analysis及适应期forecast_before_update不当作发行后的预测。',
                                '最多7个既见历史起点，不作小样本显著性或新盲测宣称。',
                                '同化等效水量逐origin的6h独立累计，不跨起点相加为连续月补水。',
                                '参数analysis与补散度两账分开；补散度不受analysis的每周期1%上限，但仍须在±5%与原物理界交集内。',
                                '输入轮廓超出仅描述采样状态，原39水量门槛失败和38全线失败保持。'],
                   baseline_replaced=False, engineering_acceptance=False,
                   summary_script_sha256=sha(Path(__file__)))
    dump(HERE / '条件性在线对照汇总.json', summary)
    dump(HERE / '汇总状态.json', status)
    print(json.dumps(dict(status=summary['status'], completed=status['completed_count'], failed=status['failed_terminal_count'], common_origins=common_issues), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
