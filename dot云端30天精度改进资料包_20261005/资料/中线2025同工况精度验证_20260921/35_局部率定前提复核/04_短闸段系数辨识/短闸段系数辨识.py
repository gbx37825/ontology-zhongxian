"""按现有位置映射，在短闸段固定n、辨识公共C倍率；不改原始观测。"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq

O = Path(__file__).resolve().parent
P = O.parent
R = P.parent
B = R / '26_冻结基线与后续归因'
sys.path.insert(0, str(P / '03_局部模型'))
sys.path.insert(0, str(R / '23_五十成员参数同化/脚本'))
from 局部闸段 import build, mean_forcing, observed, steady, sample, advance, STATE_KEYS
from 常驻水力进程 import jit_evidence

p = argparse.ArgumentParser()
p.add_argument('--case', required=True, choices=['JZZ6_short', 'JZZ16_short'])
p.add_argument('--prerequisite', required=True, help='已由主进程审查的本case前提核验文件')
a = p.parse_args()
out = O / a.case
out.mkdir(exist_ok=False)
beg = time.time()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
prerequisite = json.loads(Path(a.prerequisite).read_text())
assert prerequisite['case'] == a.case and prerequisite['eligible_for_conditional_C_trial']
station = a.case.split('_')[0]
window = pd.read_csv(P / '02_短闸段窗口/固定选择训练窗口.csv').set_index('station').loc[station]
lo, hi = float(window.start_s), float(window.end_s)
assert window.passed and hi <= 1296000
times, y = observed(station, 'forebay_m')
mask = (times >= lo) & (times <= hi)
target = float(np.trapezoid(y[mask], times[mask]) / (hi - lo))
bounds = json.loads((B / 'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())['parameter_bounds']
bound = next(v for v in bounds if v['name'] == station + '_C')
lower, upper = float(bound['lower']), float(bound['upper'])
assert lower < 1 <= upper
local = build(a.case, t0=hi)
hold = mean_forcing(local, lo, hi)
base_C = local['model']._Co.copy()
base_n = local['model']._n_ik.copy()
template = copy.deepcopy(local)
trace = []
cache = {}


def contour(local):
    m = local['model']
    meta = local['converted']['channel_discretization']
    excess = [float(m.h_Ik[v['junction_id']] - v['height']) for q in meta.values() for v in q['sections']]
    return float(max(excess))


def evaluate(factor):
    key = float(factor)
    if key in cache:
        return cache[key]
    test = copy.deepcopy(template)
    test['model']._Co[:] = base_C * factor
    assert np.array_equal(test['model']._n_ik, base_n)
    result = steady(test, kw=hold)
    if not result['converged']:
        raise RuntimeError('C辨识固定边界未收敛: ' + str(result['status']))
    state = sample(test)
    residual = state[station + '_forebay_m'] - target
    entry = dict(factor=float(factor), C_values=(base_C * factor).tolist(), predicted_forebay=state[station + '_forebay_m'], target_forebay=target, residual_m=float(residual), maximum_depth_above_input_contour_m=contour(test), steady=result)
    trace.append(entry)
    (out / '系数辨识逐次记录.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    cache[key] = (float(residual), test)
    return cache[key]


config = dict(case=a.case, station=station, pid=os.getpid(), train_window=[lo, hi], target_time_weighted_mean_m=target, C_factor_bounds=[lower, upper], original_C=base_C.tolist(), roughness_fixed=True, no_assimilation=True, observations_used_for_evaluation=['forebay_m'], tailwater_is_boundary_not_scored=True, first15_end_s=1296000, end_s=2592000, source_hashes={str(path): sha(path) for path in [B/'共享输入/C07.json', B/'共享输入/作者已核实修正观测层.csv', P/'03_局部模型/局部闸段.py', Path(__file__), Path(a.prerequisite)]}, note='条件性有效C倍率；不证明现场真实流量系数，不代表全线原模型精度。已知历史边界回放，后15天未用于选系数。')
(out / '运行约定.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))

try:
    f0, _ = evaluate(1.)
    # 固定Q及尾水时，先沿可降低当前水位残差的C方向找根，避免无意义地试过低C。
    bracket = (1., upper) if f0 > 0 else (lower, 1.)
    fl, _ = evaluate(bracket[0])
    fu, _ = evaluate(bracket[1])
    if fl * fu > 0:
        result = dict(status='NO_BRACKET_IN_SELECTED_DIRECTION', baseline_residual_m=f0, residual_at_bracket_low_m=fl, residual_at_bracket_high_m=fu, bracket=list(bracket), bounds_unchanged=True, reason='在原界内沿预期降残差方向未包围目标，不放宽或夹到边界冒称辨识成功，也不证明整个参数域无解。', wall_seconds=time.time()-beg)
    else:
        factor = float(brentq(lambda z: evaluate(z)[0], *bracket, xtol=1e-7, rtol=1e-7, maxiter=30))
        residual, fitted = evaluate(factor)
        interior = lower + 1e-5 < factor < upper - 1e-5
        if contour(fitted) > 1e-6 or not interior:
            result = dict(status='CALIBRATION_DOMAIN_FAILED', factor=factor, residual_m=residual, maximum_depth_above_input_contour_m=contour(fitted), interior=interior, wall_seconds=time.time()-beg)
        else:
            cases = []
            for label, cf in [('原系数', 1.), ('辨识系数', factor)]:
                mlocal = build(a.case, t0=0., c_factor=cf)
                init = steady(mlocal, t=0.)
                if not init['converged']:
                    raise RuntimeError(label + '起点稳态未收敛')
                (out/(label+'_起点稳态.json')).write_text(json.dumps(init, ensure_ascii=False, indent=2))
                rows = []
                retries = 0
                maximum_excess = contour(mlocal)
                for day in range(30):
                    part, retried = advance(mlocal, day*86400., (day+1)*86400.)
                    retries += retried
                    rows.extend(part)
                    maximum_excess = max(maximum_excess, contour(mlocal))
                    status = dict(status='RUNNING', branch=label, cycles=len(rows), total_cycles=1440, wall_seconds=time.time()-beg)
                    (out/'进度.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                model = mlocal['model']
                assert np.array_equal(model._Co, base_C*cf) and np.array_equal(model._n_ik, base_n)
                assert all(np.isfinite(getattr(model,k)).all() for k in STATE_KEYS)
                table = pd.DataFrame(rows)
                assert np.array_equal(table.time_s.to_numpy(), np.arange(1800,2592001,1800))
                table['observed_forebay_m'] = np.interp(table.time_s, times, y)
                table['error_m'] = table[station+'_forebay_m'] - table.observed_forebay_m
                table.to_csv(out/(label+'_30天水位.csv'),index=False)
                np.savez_compressed(out/(label+'_末态.npz'), **{k:getattr(model,k) for k in STATE_KEYS}, C=model._Co, n=model._n_ik)
                summary = dict(branch=label, factor=cf, cycles=1440, final_time_s=float(model.t), retries=retries, parameters_fixed=True, maximum_depth_excess_at_daily_check_m=maximum_excess, runtime_audit=mlocal.get('runtime_audit',{}), numba=jit_evidence(), intervals=[])
                selections = [('辨识窗', table.time_s.between(lo,hi,inclusive='both')), ('前15天非辨识窗',(table.time_s<=1296000)&~table.time_s.between(lo,hi,inclusive='both')), ('后15天',table.time_s>1296000), ('全30天',table.time_s>0)]
                for period, selected in selections:
                    err=table.loc[selected,'error_m']
                    summary['intervals'].append(dict(period=period,n=len(err),MAE=float(err.abs().mean()),RMSE=float(np.sqrt((err**2).mean())),bias=float(err.mean()),max_abs=float(err.abs().max())))
                cases.append(summary)
            original = next(v for v in cases[0]['intervals'] if v['period']=='后15天')
            calibrated = next(v for v in cases[1]['intervals'] if v['period']=='后15天')
            needed = max(.002,.05*original['MAE'])
            useful = original['MAE']-calibrated['MAE'] >= needed and calibrated['RMSE'] <= original['RMSE'] and all(v['maximum_depth_excess_at_daily_check_m']<=1e-6 for v in cases)
            assert all(sha(path)==v for path,v in config['source_hashes'].items())
            result = dict(status='COMPLETED', factor=factor, fitted_C=(base_C*factor).tolist(), calibration_residual_m=residual, runs=cases, after15_MAE_improvement_m=original['MAE']-calibrated['MAE'], predeclared_minimum_improvement_m=needed, useful_for_further_diagnostic=bool(useful), baseline_replaced=False, source_hashes_unchanged=True, wall_seconds=time.time()-beg)
    (out/'结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out/'进度.json').write_text(json.dumps(dict(status=result['status'],wall_seconds=time.time()-beg),ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
except BaseException as ex:
    (out/'失败.json').write_text(json.dumps(dict(error=repr(ex),wall_seconds=time.time()-beg),ensure_ascii=False,indent=2))
    raise
