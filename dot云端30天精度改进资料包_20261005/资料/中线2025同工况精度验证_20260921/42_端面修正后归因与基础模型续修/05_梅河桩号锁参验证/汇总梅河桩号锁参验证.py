"""梅河源桩号单项修正的只读汇总；仅真实完整终态计分，不运行水力模型。"""
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
spec = importlib.util.spec_from_file_location('_42_readonly_statistics10', SCRIPT10)
stats10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats10)
stats07 = stats10.stats07
require, read, sha, dump, records = stats10.require, stats10.read, stats10.sha, stats10.dump, stats10.records
CASES = {'reference': PAIR10 / 'reference', 'chainage': HERE / 'chainage'}
PROTOCOL_SHA256 = '81697f71e8e501410be75625025be7b5e8d8db831c42dff8a77566b6534dd820'


def reader(name, case, *args):
    # 只调整导入模块的读入根目录；不调用其旧汇总writer，不改41任何文件。
    previous = stats10.HERE
    try:
        stats10.HERE = CASES[case].parent
        return getattr(stats10, name)(case, *args)
    finally:
        stats10.HERE = previous


def identity_audit(configs, protocol):
    ref, new = configs['reference'], configs['chainage']
    for case, cfg in configs.items():
        stats07.fingerprints(cfg)
        require(cfg['parameters_fixed'] and not cfg['assimilation'] and cfg['state_corrections'] == 0,
                '存在调参、同化或状态校正')
        require(not cfg['endpoint_hook_applied'] and cfg['endpoint_geometry_manifest'] is None,
                '本轮不允许端面hook')
        for key in ['flow_tol_m3s', 'head_tol_m', 'num_iter', 'strict', 'dt_s', 'minimum_failed_dt_s']:
            require(cfg['protocol'][key] == protocol[key], '数值设置不一致: ' + case + '/' + key)
        with np.load(CASES[case] / '实际初始化_0s.npz') as z:
            require(set(z.files) == set(stats07.STATE_KEYS), '保存初态应只含六态，不含旧几何缓存')
            require(all(stats07.arrsha(z[k]) == ref['initial_state_sha256'][k] for k in stats07.STATE_KEYS),
                    '实际初态不等于reference')
    require(new['protocol'] == protocol, '候选没有使用冻结的05约定')
    for key in ['initial_state_sha256', 'parameter_sha256', 'forcing_array_sha256']:
        require(new[key] == ref[key], '六态、原参数或五个驱动数组指纹不同: ' + key)
    for key in ['primary_water_groups', 'secondary_water_groups', 'flow_groups', 'score_metrics',
                'numeric_comparison_tolerance', 'conditional_accuracy_gate']:
        require(protocol['accuracy_assessment'][key] == ref['protocol']['accuracy_assessment'][key],
                '精度口径没有沿用41/10: ' + key)
    require(new['source_geometry_changed'] and new['internal_length_changed'] and
            not new['internal_midpoint_cross_section_changed'], '候选应仅修改源长度，不能套端面几何身份')
    manifest = read(CASES['chainage'] / '源几何与算子身份核验.json')
    require(manifest == new['chainage_manifest'], '实际候选manifest与配置不同')
    require(manifest['only_internal_length_change'] == {'id': 939, 'old_m': 313., 'new_m': 43.},
            '首连接长度改动不符')
    for key in ['reference_arrays_hash_verified', 'water_operators_43_exact', 'flow_operators_44_exact',
                'offtake_routes_128_exact', 'forcing_five_arrays_exact']:
        require(manifest[key] is True, '算子/驱动/原参照数组核验未通过: ' + key)
    for key in ['reference_x_cache_loaded', 'endpoint_hook', 'gate_bottom_hook',
                'western_connection_repair', 'S02_storage_candidate']:
        require(manifest[key] is False, '混入其他候选或旧x缓存: ' + key)

    source = read(ROOT / '26_冻结基线与后续归因/共享输入/C07.json')
    candidate_path = CASES['chainage'] / '候选源输入.json'
    candidate = read(candidate_path)
    require(sha(candidate_path) == new['candidate_source_file_sha256'], '候选源文件指纹不同')
    require(source['中线全线']['baseData']['sections']['q20s1']['location'] == 385036 and
            candidate['中线全线']['baseData']['sections']['q20s1']['location'] == 385306, '源桩号不符')
    restored = copy.deepcopy(candidate)
    restored['中线全线']['baseData']['sections']['q20s1']['location'] = 385036
    require(restored == source, '除q20s1桩号外还修改了其他源值')

    with np.load(CASES['chainage'] / '参考参数与内部几何数组.npz') as old, \
         np.load(CASES['chainage'] / '候选构造几何与连接数组.npz') as candidate_arrays:
        for key, digest in {**ref['parameter_sha256'], **ref['internal_geometry_sha256']}.items():
            require(stats07.arrsha(old[key]) == digest, '保存原参照数组与reference指纹不同: ' + key)
        for key, digest in new['internal_geometry_sha256'].items():
            require(stats07.arrsha(candidate_arrays[key]) == digest, '候选几何数组指纹不同: ' + key)
            if key != '_dx_ik':
                require(np.array_equal(old[key], candidate_arrays[key]), '非内部长度几何数组改变: ' + key)
        before, after = old['_dx_ik'], candidate_arrays['_dx_ik']
        changed = np.flatnonzero(before != after).tolist()
        require(changed == [939] and before[939] == 313 and after[939] == 43, '不止dx939改变')
        x, left, right = candidate_arrays['_x_Ik'], candidate_arrays['_Ik'], candidate_arrays['_Ip1k']
        require(np.max(abs(x[right]-x[left]-after)) < 1e-8, '候选x坐标与新dx不一致')
        ids = np.flatnonzero(candidate_arrays['_ki'] == 38)
        require(left[939] == 977 and x[977] == 0 and x[right[939]] == 43,
                '梅河首段没有使用候选自身构造坐标')
        channel_length = float(after[ids].sum())
        require(abs(channel_length-17117.) < 1e-8, '梅河候选总长不符')

    water_ref = read(CASES['reference'] / '实际位置观测算子.json')
    require(len(water_ref) == 43 and water_ref == read(CASES['chainage'] / '实际位置观测算子.json'),
            '43安装位置算子改变')
    flow_ref = read(CASES['reference'] / '测流位置算子与适用范围.json')
    flow_new = read(CASES['chainage'] / '测流位置算子与适用范围.json')
    require(len(flow_ref['operators']) == 44 and all(flow_ref[k] == flow_new[k] for k in ['operators', 'audit']),
            '44测流算子或近似口径改变')
    routes = read(CASES['chainage'] / '分水路由身份.json')
    require(routes['unchanged_against_original_source'] and len(routes['routes']) == 128, '128分水路由核验不符')
    require(read(CASES['reference'] / '原始与端面轮廓定义.json') ==
            read(CASES['chainage'] / '原始与端面轮廓定义.json'), '固定输入轮廓定义改变')
    volume = read(CASES['chainage'] / '初始几何容积差.json')
    delta = new['initial_geometric_volume_m3']-ref['initial_geometric_volume_m3']
    expected = protocol['initial_volume_difference']['candidate_minus_reference_expected_m3']
    require(abs(delta-expected) < 1e-5 and abs(delta-volume['candidate_minus_reference_m3']) < 1e-8 and
            abs(delta+270*volume['first_link_area_m2']) < 1e-5, '初始体积差不等于首段长度差乘原初态面积')
    require(volume['not_a_dynamic_water_balance_residual'] and not volume['artificial_water_compensation'],
            '初始几何体积差被当作运行残差或人工补水')
    return dict(source_only_change='/中线全线/baseData/sections/q20s1/location:385036→385306',
                changed_internal_length_ids=changed, old_first_length_m=313., new_first_length_m=43.,
                candidate_channel_length_m=channel_length, candidate_x_matches_own_dx=True,
                other_internal_geometry_arrays_exact=True, six_initial_states_exact=True,
                original_parameters_exact=True, five_forcing_array_hashes_exact=True,
                water_operators_43_exact=True, flow_operators_44_exact=True,
                offtake_routes_128_verified_by_runtime_manifest=True, runtime_manifest=manifest,
                initial_geometric_volume_difference=volume, initial_volume_difference_separate_from_dynamic_balance=True,
                reference_reused_without_rerun=True, endpoint_hook=False, assimilation=False)


def write_accuracy(frames, protocol):
    overall, points = stats07.score_tables(frames, include_additional_groups=True)
    paired = stats07.changes_table(points, 'reference', 'chainage')
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
    changes = overall[overall.case == 'chainage'].merge(overall[overall.case == 'reference'],
        on=keys, suffixes=('_chainage', '_reference'), validate='one_to_one')
    for key in ['MAE', 'RMSE', 'bias', 'max_abs']:
        changes['delta_'+key] = changes[key+'_chainage']-changes[key+'_reference']
    checks = {}
    for period, _, _ in stats07.PERIODS:
        use = paired[(paired.period == period) & paired.score_group.isin(['118闸节点水位', '43安装位置水位'])]
        require(len(use) == 161, '全部161种水位表示没有配对齐全')
        checks[period] = dict(maximum_increase_m=float(use.delta_MAE.max()),
                             worst_point=records(use.loc[[use.delta_MAE.idxmax()]])[0],
                             distinct_water_representations_over_1cm=int((use.delta_MAE > .01).sum()))
    # 复用41的门槛函数；其旧字段名仅在函数调用中适配，不写成端面候选结果。
    assessment_rows = [dict(row, MAE_endpoint=row['MAE_chainage']) for row in records(changes)]
    gate = stats10.assess_accuracy(assessment_rows, checks, protocol)
    gate['candidate_case'] = 'chainage'
    for filename, frame in [('梅河桩号锁参_分组精度.csv', overall), ('梅河桩号锁参_逐测点精度.csv', points),
                            ('梅河桩号锁参_逐点变化.csv', paired),
                            ('梅河桩号锁参_水位MAE退化超过1cm.csv', paired[paired.water_MAE_worse_over_1cm])]:
        frame.to_csv(HERE / filename, index=False)
    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata[metadata.kind == 'meters'][['case', 'station', 'chainage_m', 'source_name',
        'direct_forcing_observation', 'end_connection_approximation', 'end_connection_offset_m']].drop_duplicates()
    metadata.to_csv(HERE / '44测流输入来源与端部近似.csv', index=False)
    return dict(group_accuracy=records(overall), group_changes=records(changes), water_point_checks=checks,
                water_point_MAE_worse_over_1cm=records(paired[paired.water_MAE_worse_over_1cm]),
                conditional_accuracy_gate=gate)


def summarize(check_only):
    statuses = {case: stats10.run_status(folder) for case, folder in CASES.items()}
    if not all(s['terminal'] for s in statuses.values()):
        return dict(status='WAITING_FOR_TRUE_TERMINAL', cases=statuses, precision_results_written=False)
    if not all(s['status'] == 'COMPLETED' and s['completed_cycles'] == 1440 for s in statuses.values()):
        result = dict(status='INCOMPLETE_OR_FAILED_NO_FULL_MONTH_SCORE', cases=statuses,
                      partial_results_scored=False, precision_results_written=False, baseline_replaced=False)
        if not check_only:
            dump(HERE / '梅河桩号锁参验证结论.json', result)
        return result
    protocol_path = HERE / '运行前约定.json'
    require(sha(protocol_path) == PROTOCOL_SHA256, '运行前约定指纹改变')
    protocol = read(protocol_path)
    configs = {case: read(folder / '运行配置.json') for case, folder in CASES.items()}
    identity = identity_audit(configs, protocol)
    reference_exact = stats10.reference_reproduction()
    saved_exact = read(CASES['reference'] / '与06逐值复现核验.json')
    require(saved_exact['all_exact'] and all(saved_exact['checks'].values()), 'reference既有逐值核验失败')
    observations = stats07.observations()
    membership = stats07.numerical_group_membership(observations)
    frames, local_water, global_water, domains, means = [], [], [], [], {}
    case_checks, definitions, sources = {}, {}, {}
    for case, folder in CASES.items():
        frame = reader('load_case_scores', case, observations)
        instant, ledger, means[case], chunks = reader('lanhe_water', case)
        meter = frame[(frame.kind == 'meters') & (frame.station == 'JZZ16')].sort_values('time_s')
        require(np.max(abs(instant.sim.to_numpy()-meter.sim.to_numpy())) < 1e-10, '44测流中的Q16不等于逐步末态')
        water, domain, definitions[case] = reader('domain_and_network_steps', case, configs[case])
        case_checks[case] = reader('final_result_checks', case, ledger, chunks)
        case_checks[case]['chunks'] = chunks
        frames.append(frame)
        local_water += ledger
        global_water += water
        domains += domain
        sources.update(configs[case]['source_hashes'])
        sources.update({str(p): sha(p) for p in folder.iterdir() if p.suffix in ['.json', '.csv', '.npz', '.jsonl']})
    temporal = stats10.q16_temporal_scores(frames, observations, means)
    sources.update(membership['source_sha256'])
    for path in [SCRIPT10, stats10.REUSED_SCRIPT, Path(__file__), protocol_path]:
        sources[str(path)] = sha(path)
    result = dict(status='COMPLETE_CHAINAGE_LOCKED_PARAMETER_PAIRED_DESCRIPTION', cases=statuses,
        identity=identity, reference_exact_to_06=reference_exact, score_group_membership=membership,
        runtime_checks=case_checks, Q16_temporal_scores=temporal, lanhe_water=local_water,
        whole_network_water=global_water, contour_definitions_and_initial_ranges=definitions,
        source_sha256=sources, precision_results_written=not check_only,
        baseline_replaced=False, engineering_acceptance=False,
        limitations=['唯一源改动为梅河q20s1桩号；未合入端面、闸底、西赵河、S02候选。',
            '相同六态配新长度的初始几何体积少50619.831m3，单列；动态水量账各自从本分支初始体积起算。',
            '44测流保留驱动来源重合及JZZ17/59端部近似标志；其余42不等于全部现场身份已确认。',
            'Q16末态误差与半小时积分均流分开，观测均流是相邻半小时端点线性均值。',
            '全网与兰河控制体和残差符号不同；几何减离散蓄变差不等于现场水损。',
            '固定输入轮廓域外不是现场漫顶；源桩号纠正、精度门槛、物理验收及基线替换分别判断。'])
    if not check_only:
        result.update(write_accuracy(frames, protocol))
        for name, rows in [('Q16末态与半小时均流分别计分.csv', temporal), ('兰河离散水量账.csv', local_water),
                           ('全网离散与几何储水账.csv', global_water), ('原始与端面轮廓域外描述.csv', domains)]:
            pd.DataFrame(rows).to_csv(HERE / name, index=False)
        outputs = {str(p): sha(p) for p in HERE.glob('*.csv')}
        result['output_csv_sha256'] = outputs
        dump(HERE / '梅河桩号锁参验证结论.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='只核验完整终态及身份，不写精度结果')
    args = parser.parse_args()
    try:
        result = summarize(args.check_only)
    except Exception as exc:
        print(json.dumps(dict(status='BLOCKED_BY_ARTIFACT_OR_IDENTITY_CHECK', error=repr(exc),
                              precision_results_written=False), ensure_ascii=False))
        return 2
    print(json.dumps({k: result[k] for k in ['status', 'cases', 'precision_results_written']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
