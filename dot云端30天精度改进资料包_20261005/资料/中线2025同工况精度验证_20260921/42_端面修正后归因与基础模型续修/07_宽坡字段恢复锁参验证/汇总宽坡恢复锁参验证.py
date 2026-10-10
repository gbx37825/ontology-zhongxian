"""两个固定宽坡恢复候选分别对同一冻结参考；只读汇总，不运行水力模型。"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PAIR10 = ROOT / '41_基础模型深入检查与修复/10_端面几何锁参验证'
SCRIPT10 = PAIR10 / '汇总端面几何锁参验证.py'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('_42_width_slope_statistics10', SCRIPT10)
stats10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats10)
stats07 = stats10.stats07
require, read, sha, dump, records = stats10.require, stats10.read, stats10.sha, stats10.dump, stats10.records
CANDIDATES = ('q22', 'q26')
FOLDERS = {'reference': PAIR10 / 'reference', **{case: HERE / case for case in CANDIDATES}}
# 主进程审查的最终约定；不是根据后续结果调整的门槛。
PROTOCOL_SHA256 = '831ce073fe67c7d5a8522913c993188c36f305f787c6b98b738fe4869fa1b731'


def reader(name, case, *args):
    previous = stats10.HERE
    try:
        stats10.HERE = FOLDERS[case].parent
        return getattr(stats10, name)(case, *args)
    finally:
        stats10.HERE = previous


def identity_audit(case, configs, protocol):
    ref, new, plan = configs['reference'], configs[case], protocol['cases'][case]
    folder = FOLDERS[case]
    expected_ids = [1067, 1068] if case == 'q22' else list(range(1187, 1194))
    expected_sections = ['q22s14'] if case == 'q22' else ['q26s1', 'q26s2', 'q26s3', 'q26s4']
    require(plan['allowed_changed_internal_g2_g3_link_ids'] == expected_ids and
            plan['source_sections'] == expected_sections, '固定候选范围不符')
    require(sha(plan['patch_path']) == plan['patch_sha256'] and
            sha(plan['zero_step_audit_path']) == plan['zero_step_audit_sha256'], '候选或零步核验指纹改变')
    patch, zero = read(plan['patch_path']), read(plan['zero_step_audit_path'])
    require(zero['hydraulic_steps'] == 0 and [r['id'] for r in zero['changed_g2_g3_links']] == expected_ids,
            '零步核验的候选身份不符')
    for name, cfg in [('reference', ref), (case, new)]:
        stats07.fingerprints(cfg)
        require(cfg['parameters_fixed'] and not cfg['assimilation'] and cfg['state_corrections'] == 0,
                '存在调参、同化或状态校正')
        require(not cfg['endpoint_hook_applied'] and cfg['endpoint_geometry_manifest'] is None,
                '本轮不允许端面hook')
        for key in ['flow_tol_m3s', 'head_tol_m', 'num_iter', 'strict', 'dt_s', 'minimum_failed_dt_s']:
            require(cfg['protocol'][key] == protocol[key], '数值设置不一致: '+name+'/'+key)
        with np.load(FOLDERS[name] / '实际初始化_0s.npz') as z:
            require(set(z.files) == set(stats07.STATE_KEYS), '保存初态应只含六态，不含旧几何缓存')
            require(all(stats07.arrsha(z[k]) == ref['initial_state_sha256'][k] for k in stats07.STATE_KEYS),
                    '实际六态不等于冻结参考')
    require(new['protocol'] == protocol, '候选没有使用冻结07约定')
    for key in ['initial_state_sha256', 'parameter_sha256', 'forcing_array_sha256']:
        require(new[key] == ref[key], '初态、原参数或五个驱动数组指纹不同: '+key)
    for key in ['primary_water_groups', 'secondary_water_groups', 'flow_groups', 'score_metrics',
                'numeric_comparison_tolerance', 'conditional_accuracy_gate']:
        require(protocol['accuracy_assessment'][key] == ref['protocol']['accuracy_assessment'][key],
                '精度口径没有沿用41/10: '+key)
    require(new['source_geometry_changed'] and new['internal_midpoint_cross_section_changed'] and
            not new['internal_length_changed'], '本候选只能改变宽坡，不能混入长度或端面hook候选')
    manifest = read(folder / '源几何与算子身份核验.json')
    require(manifest == new['width_slope_manifest'] and manifest['case'] == case and
            manifest['source_sections'] == expected_sections and
            manifest['only_g2_g3_changed_link_ids'] == expected_ids, '实际manifest与固定候选不符')
    for key in ['all_other_geometry_and_connections_exact', 'x_Ik_exact_reference',
                'reference_arrays_hash_verified', 'closed_105_end_geometry_exact', 'water_operators_43_exact',
                'flow_operators_44_exact', 'offtake_routes_128_exact', 'forcing_five_arrays_exact']:
        require(manifest[key] is True, '几何/算子/驱动核验未通过: '+key)
    for key in ['reference_x_cache_loaded', 'q20_chainage_candidate', 'endpoint_hook', 'gate_bottom_hook',
                'western_connection_repair', 'S02_storage_candidate']:
        require(manifest[key] is False, '混入其他候选或旧x缓存: '+key)

    original_path = ROOT / '26_冻结基线与后续归因/共享输入/C07.json'
    original = read(original_path)
    require(patch['base_sha256'] == sha(original_path) and patch['patch_root'] == '中线全线',
            '补丁根或来源不符')
    expected_source = copy.deepcopy(original)
    require([r['path'] for r in patch['patches']] == plan['allowed_changed_source_fields'], '补丁字段越界')
    for row in patch['patches']:
        require(row['op'] == 'replace', '补丁必须只替换固定横坐标')
        tokens = row['path'].strip('/').split('/')
        target = expected_source['中线全线']
        for token in tokens[:-1]:
            target = target[int(token)] if isinstance(target, list) else target[token]
        key = int(tokens[-1]) if isinstance(target, list) else tokens[-1]
        require(target[key] == row['old'], '补丁原值不符')
        target[key] = row['value']
    candidate_path = folder / '候选源输入.json'
    require(read(candidate_path) == expected_source and sha(candidate_path) == new['candidate_source_file_sha256'],
            '除固定宽坡横坐标外还改变了源输入')

    fixed_extended = ['_x_Ik', '_Ik', '_Ip1k', '_ki', '_J_uk', '_J_dk', '_J_uo', '_J_do',
                      '_z_inv_uk', '_z_inv_dk', '_z_o', '_geom_codes']
    with np.load(folder / '参考参数与内部几何数组.npz') as old, \
         np.load(folder / '候选构造几何与连接数组.npz') as arrays:
        for key, digest in {**ref['parameter_sha256'], **ref['internal_geometry_sha256']}.items():
            require(stats07.arrsha(old[key]) == digest, '保存参考数组与冻结参考不同: '+key)
        for key, digest in new['reference_geometry_sha256'].items():
            require(stats07.arrsha(old[key]) == digest, '参考重建几何指纹不符: '+key)
        require(set(new['internal_geometry_sha256']) == set(ref['internal_geometry_sha256']) | set(fixed_extended),
                '候选几何/纵向坐标/连接锁定数组不完整')
        changed = {}
        for key, digest in new['internal_geometry_sha256'].items():
            require(stats07.arrsha(arrays[key]) == digest, '候选实际几何数组指纹不符: '+key)
            ids = np.flatnonzero(old[key] != arrays[key]).tolist()
            if key in ['_g2_ik', '_g3_ik']:
                require(ids == expected_ids, '宽坡数组改变位置越界: '+key)
                changed[key] = ids
            else:
                require(not ids, '宽坡以外的数组改变: '+key)
        for row in zero['changed_g2_g3_links']:
            i = row['id']
            require(old['_g2_ik'][i] == row['old_g2_m'] and arrays['_g2_ik'][i] == row['new_g2_m'] and
                    old['_g3_ik'][i] == row['old_g3'] and arrays['_g3_ik'][i] == row['new_g3'] and
                    arrays['_dx_ik'][i] == row['length_m'] and arrays['_g1_ik'][i] == row['g1_m'],
                    '实际变化值没有逐值复现零步核验')
        x, left, right = arrays['_x_Ik'], arrays['_Ik'], arrays['_Ip1k']
        require(np.max(abs(x[right]-x[left]-arrays['_dx_ik'])) < 1e-8, '纵向坐标与固定长度不自洽')

    water_ref = read(FOLDERS['reference'] / '实际位置观测算子.json')
    require(len(water_ref) == 43 and read(folder / '实际位置观测算子.json') == water_ref, '43水位算子改变')
    flow_ref = read(FOLDERS['reference'] / '测流位置算子与适用范围.json')
    flow_new = read(folder / '测流位置算子与适用范围.json')
    require(len(flow_ref['operators']) == 44 and all(flow_ref[k] == flow_new[k] for k in ['operators', 'audit']),
            '44测流算子或端部近似口径改变')
    routes = read(folder / '分水路由身份.json')
    require(routes['unchanged_against_original_source'] and len(routes['routes']) == 128, '128分水核验不符')

    # 原算法的输入轮廓高度保持；源端面宽坡元数据可随本候选恢复，不能套05的整字典相等。
    d0 = read(FOLDERS['reference'] / '原始与端面轮廓定义.json')
    d1 = read(folder / '原始与端面轮廓定义.json')
    e0, e1 = d0.pop('endpoints'), d1.pop('endpoints')
    require(d0 == d1 and len(e0) == len(e1), '固定轮廓定义、节点集合或阈值改变')
    contour_changes = []
    point_source = {row['section']: row for row in patch['points']}
    for before, after in zip(e0, e1):
        require(before.keys() == after.keys(), '端面元数据字段变化')
        fields = [key for key in before if before[key] != after[key]]
        if fields:
            require(after['source_section'] in expected_sections and
                    set(fields).issubset({'endpoint_bottom_width_m', 'endpoint_side_slope'}),
                    '端面高度、床高、连接身份或非候选宽坡改变')
            point = point_source[after['source_section']]
            require(abs(after['endpoint_bottom_width_m']-point['source_b_m']) < 1e-9 and
                    abs(after['endpoint_side_slope']-point['source_slope']) < 1e-9, '端面宽坡不是原表候选值')
            contour_changes.append(dict(channel=after['channel'], side=after['side'],
                source_section=after['source_section'], fields=fields,
                before={key: before[key] for key in fields}, after={key: after[key] for key in fields}))
    require(read(folder / '初始轮廓范围.json') == read(FOLDERS['reference'] / '初始轮廓范围.json'),
            '同六态/同固定高度下初始轮廓范围不一致')
    volume, areas = read(folder / '初始几何容积差.json'), read(folder / '初始连接面积差.json')
    delta = new['initial_geometric_volume_m3']-ref['initial_geometric_volume_m3']
    require([row['link_id'] for row in areas['rows']] == expected_ids and
            areas['rows'] == volume['changed_link_areas'], '初始面积变化连接不符')
    independent_delta = 0.
    for row, zero_row in zip(areas['rows'], zero['changed_g2_g3_links']):
        require(abs(row['original_area_m2']-zero_row['old_initial_A_m2']) < 1e-9 and
                abs(row['candidate_area_m2']-zero_row['new_initial_A_m2']) < 1e-9 and
                row['length_m'] == zero_row['length_m'], '初始连接面积未复现零步核验')
        value = (row['candidate_area_m2']-row['original_area_m2'])*row['length_m']
        require(abs(value-row['volume_change_m3']) < 1e-8, '逐连接初始体积差算术不符')
        independent_delta += value
    require(abs(delta-plan['expected_initial_volume_difference_m3']) < 1e-5 and
            abs(delta-volume['candidate_minus_reference_m3']) < 1e-8 and
            abs(delta-independent_delta) < 1e-5 and
            abs(delta-areas['independent_volume_change_m3']) < 1e-5, '初始几何体积差没有独立闭合')
    require(areas['reference_geometry_only_reconstructed_on_copy'] and areas['reference_hydraulic_steps'] == 0 and
            volume['not_a_dynamic_water_balance_residual'] and not volume['artificial_water_compensation'],
            '参考被推进或初始体积差被作为人工补水/动态残差')
    return dict(candidate=case, source_patch=plan, changed_g2_g3_arrays=changed,
                all_other_geometry_and_connections_exact=True, six_initial_states_exact=True,
                original_parameters_exact=True, five_forcing_array_hashes_exact=True,
                water_operators_43_exact=True, flow_operators_44_exact=True,
                offtake_routes_128_verified_by_runtime_manifest=True, runtime_manifest=manifest,
                fixed_contour_heights_and_ids_exact=True, allowed_contour_width_slope_metadata_changes=contour_changes,
                initial_geometric_volume_difference=volume, initial_link_area_check=areas,
                initial_volume_difference_separate_from_dynamic_balance=True,
                reference_reused_without_rerun=True, candidates_combined=False, endpoint_hook=False, assimilation=False)


def write_accuracy(frames, protocol, candidate):
    overall, points = stats07.score_tables(frames, include_additional_groups=True)
    paired = stats07.changes_table(points, 'reference', candidate)
    supplement = []
    for frame in frames:
        flow = frame[frame.kind == 'meters']
        subsets = [('JZZ17端部最近中心近似测流', flow[flow.station == 'JZZ17'], 1),
                   ('JZZ59端部最近中心近似测流', flow[flow.station == 'JZZ59'], 1),
                   ('剔除2个已知端部近似后的42测流_描述性', flow[~flow.station.isin(['JZZ17', 'JZZ59'])], 42)]
        for period, lo, hi in stats07.PERIODS:
            for name, values, count in subsets:
                selected = values[(values.time_s > lo) & (values.time_s <= hi)]
                require(len(selected) == count*((hi-lo)//1800), '流量补充分组覆盖不足')
                supplement.append(dict(case=str(frame.case.iloc[0]), period=period, score_group=name,
                                       unit='m3/s', points=count, **stats07.stats(selected.error)))
    overall = pd.concat([overall, pd.DataFrame(supplement)], ignore_index=True)
    keys = ['period', 'score_group', 'unit', 'points', 'samples']
    changes = overall[overall.case == candidate].merge(overall[overall.case == 'reference'],
        on=keys, suffixes=('_candidate', '_reference'), validate='one_to_one')
    for key in ['MAE', 'RMSE', 'bias', 'max_abs']:
        changes['delta_'+key] = changes[key+'_candidate']-changes[key+'_reference']
    checks = {}
    for period, _, _ in stats07.PERIODS:
        use = paired[(paired.period == period) & paired.score_group.isin(['118闸节点水位', '43安装位置水位'])]
        require(len(use) == 161, '全部161种水位表示没有配对齐全')
        checks[period] = dict(maximum_increase_m=float(use.delta_MAE.max()),
                             worst_point=records(use.loc[[use.delta_MAE.idxmax()]])[0],
                             distinct_water_representations_over_1cm=int((use.delta_MAE > .01).sum()))
    # 复用41的门槛函数；其旧字段名仅在函数调用中适配，不写成端面候选结果。
    assessment_rows = [dict(row, MAE_endpoint=row['MAE_candidate']) for row in records(changes)]
    gate = stats10.assess_accuracy(assessment_rows, checks, protocol)
    gate['candidate_case'] = candidate
    for filename, frame in [(f'{candidate}_宽坡恢复_分组精度.csv', overall), (f'{candidate}_宽坡恢复_逐测点精度.csv', points),
                            (f'{candidate}_宽坡恢复_逐点变化.csv', paired),
                            (f'{candidate}_宽坡恢复_水位MAE退化超过1cm.csv', paired[paired.water_MAE_worse_over_1cm])]:
        frame.to_csv(HERE / filename, index=False)
    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata[metadata.kind == 'meters'][['case', 'station', 'chainage_m', 'source_name',
        'direct_forcing_observation', 'end_connection_approximation', 'end_connection_offset_m']].drop_duplicates()
    metadata.to_csv(HERE / f'{candidate}_44测流输入来源与端部近似.csv', index=False)
    return dict(group_accuracy=records(overall), group_changes=records(changes), water_point_checks=checks,
                water_point_MAE_worse_over_1cm=records(paired[paired.water_MAE_worse_over_1cm]),
                conditional_accuracy_gate=gate)


def load_complete_case(case, config, observations):
    folder = FOLDERS[case]
    frame = reader('load_case_scores', case, observations)
    instant, ledger, means, chunks = reader('lanhe_water', case)
    meter = frame[(frame.kind == 'meters') & (frame.station == 'JZZ16')].sort_values('time_s')
    require(np.max(abs(instant.sim.to_numpy()-meter.sim.to_numpy())) < 1e-10,
            '44测流中的Q16不等于逐步末态')
    water, domain, definition = reader('domain_and_network_steps', case, config)
    checks = reader('final_result_checks', case, ledger, chunks)
    checks['chunks'] = chunks
    sources = {**config['source_hashes'], **{str(p): sha(p) for p in folder.iterdir()
               if p.suffix in ['.json', '.csv', '.npz', '.jsonl']}}
    return dict(frame=frame, means=means, lanhe=ledger, network=water, domains=domain,
                definition=definition, checks=checks, sources=sources)


def summarize(selected, check_only):
    statuses = {case: stats10.run_status(FOLDERS[case]) for case in ['reference', *selected]}
    terminal = [case for case in selected if statuses[case]['terminal']]
    if not terminal:
        return dict(status='WAITING_FOR_TRUE_TERMINALS', cases=statuses, precision_results_written=False)
    require(PROTOCOL_SHA256 and sha(HERE / '运行前约定.json') == PROTOCOL_SHA256,
            '07最终运行前约定尚未冻结或指纹改变')
    protocol = read(HERE / '运行前约定.json')
    require(statuses['reference']['status'] == 'COMPLETED' and statuses['reference']['completed_cycles'] == 1440,
            '冻结参考没有真实完整30天终态')
    summaries = {}
    completed = [case for case in terminal if statuses[case]['status'] == 'COMPLETED'
                 and statuses[case]['completed_cycles'] == 1440]
    for case in terminal:
        if case not in completed:
            failure = dict(status='INCOMPLETE_OR_FAILED_NO_FULL_MONTH_SCORE', cases={case: statuses[case]},
                partial_results_scored=False, precision_results_written=False,
                baseline_replaced=False, engineering_acceptance=False)
            summaries[case] = failure
            if not check_only:
                dump(HERE / f'{case}_宽坡恢复锁参验证结论.json', failure)
    if completed:
        reference_config = read(FOLDERS['reference'] / '运行配置.json')
        configs = {'reference': reference_config, **{case: read(FOLDERS[case] / '运行配置.json') for case in completed}}
        identities = {case: identity_audit(case, configs, protocol) for case in completed}
        reference_exact = stats10.reference_reproduction()
        saved_exact = read(FOLDERS['reference'] / '与06逐值复现核验.json')
        require(saved_exact['all_exact'] and all(saved_exact['checks'].values()), 'reference既有逐值核验失败')
        observations = stats07.observations()
        membership = stats07.numerical_group_membership(observations)
        ref = load_complete_case('reference', reference_config, observations)
        for case in completed:
            new = load_complete_case(case, configs[case], observations)
            frames = [ref['frame'], new['frame']]
            temporal = stats10.q16_temporal_scores(frames, observations,
                                                 {'reference': ref['means'], case: new['means']})
            source_hashes = {**ref['sources'], **new['sources'], **membership['source_sha256']}
            for p in [SCRIPT10, stats10.REUSED_SCRIPT, Path(__file__), HERE / '运行前约定.json']:
                source_hashes[str(p)] = sha(p)
            result = dict(status='COMPLETE_WIDTH_SLOPE_LOCKED_PARAMETER_PAIRED_DESCRIPTION', candidate=case,
                cases={key: statuses[key] for key in ['reference', case]}, identity=identities[case],
                reference_exact_to_06=reference_exact, score_group_membership=membership,
                runtime_checks={'reference': ref['checks'], case: new['checks']},
                Q16_temporal_scores=temporal, lanhe_water=ref['lanhe']+new['lanhe'],
                whole_network_water=ref['network']+new['network'],
                contour_definitions_and_initial_ranges={'reference': ref['definition'], case: new['definition']},
                source_sha256=source_hashes, precision_results_written=not check_only,
                independent_against_same_reference=True, candidates_combined=False, selected_as_best=False,
                baseline_replaced=False, engineering_acceptance=False,
                limitations=['两个固定候选各自对41/10 reference，不合并、不筛选日期或测点、不按结果选优。',
                    '六态固定时宽坡改变造成初始几何体积差，单列；动态水量账各自从本分支初始体积起算。',
                    '44测流保留驱动来源及JZZ17/59端部近似；其余42不等于全部现场测点身份已确认。',
                    'Q16末态误差与半小时积分均流分开；观测均流是相邻半小时端点线性均值。',
                    '全网与兰河控制体和残差符号不同；几何减离散蓄变差不等于现场水损。',
                    '固定输入轮廓域外不是现场漫顶；源恢复、精度门槛、物理验收及基线替换分别判断。'])
            if not check_only:
                result.update(write_accuracy(frames, protocol, case))
                for name, rows in [('Q16末态与半小时均流分别计分.csv', temporal),
                    ('兰河离散水量账.csv', result['lanhe_water']), ('全网离散与几何储水账.csv', result['whole_network_water']),
                    ('原始与端面轮廓域外描述.csv', ref['domains']+new['domains'])]:
                    pd.DataFrame(rows).to_csv(HERE / (case+'_'+name), index=False)
                result['output_csv_sha256'] = {str(p): sha(p) for p in HERE.glob(case+'_*.csv')}
                dump(HERE / f'{case}_宽坡恢复锁参验证结论.json', result)
            summaries[case] = dict(status=result['status'], precision_results_written=result['precision_results_written'])
    return dict(status='INDEPENDENT_CANDIDATE_TERMINALS_REVIEWED', cases=statuses, candidates=summaries,
                precision_results_written=any(x['precision_results_written'] for x in summaries.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CANDIDATES], default='all')
    parser.add_argument('--check-only', action='store_true', help='只读完整终态及身份，不写任何精度结果')
    args = parser.parse_args()
    selected = CANDIDATES if args.case == 'all' else (args.case,)
    try:
        result = summarize(selected, args.check_only)
    except Exception as exc:
        print(json.dumps(dict(status='BLOCKED_BY_ARTIFACT_OR_IDENTITY_CHECK', error=repr(exc),
                              precision_results_written=False), ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
