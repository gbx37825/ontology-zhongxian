"""固定已辨识候选，仅复用原A的T04两种对照；不拟合、不重跑基线或全30天。"""
from pathlib import Path
from collections import Counter
import copy,hashlib,json,sys,time,traceback
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;P=O.parents[1];R=P.parent
sys.path.insert(0,str(P/'01_局部模型'))
import 测点假设局部模型 as api

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
fit_path=P/'04_分通道联合辨识/A/辨识结果.json'
contract_path=P/'01_局部模型/前提核验约定.json'
fit=json.loads(fit_path.read_text());old_contract=json.loads(contract_path.read_text())
assert fit['status']=='FITTED' and fit['hypothesis']=='A'
factors=fit['ratios'];windows=old_contract['windows']
assert windows==[[216000.,237600.],[648000.,669600.]]
assert old_contract['head_tolerance_m']==.03 and old_contract['relative_gate_flow_tolerance']==.02
assert sha(api.__file__)==old_contract['api_hash']
if (O/'复核结果.json').exists():raise RuntimeError('已存在终态结果，拒绝覆盖')
paths=[Path(__file__),Path(api.__file__),fit_path,contract_path,P/'01_局部模型/两假设双初值与T04.py',R/'26_冻结基线与后续归因/共享输入/C07.json',R/'26_冻结基线与后续归因/共享输入/作者已核实修正观测层.csv',R/'26_冻结基线与后续归因/共享输入/2025开度归一化上限.json',R/'26_冻结基线与后续归因/共享输入/Pipedream闸流系数.json']
source_hashes={str(p):sha(p) for p in paths}
for module in fit['runtime_identity']['modules']:
 assert sha(module['file'])==module['sha256'];source_hashes[module['file']]=module['sha256']
settings=dict(hypothesis='A',fixed_ratios=factors,effective_n=fit['effective_n_min'],effective_C=fit['effective_C'],windows=windows,head_limit_m=.03,gate_relative_limit=.02,method='复用原T04：末时固定稳态vs提前24h到达末态；窗均值固定稳态vs真实最后6h时间加权均值',time_step_branch_comparison=False,baseline_rerun=False,full_30day_rerun=False,parameter_tuning=False,source_sha256=source_hashes)
dump(O/'本次复核约定.json',settings)

def build(t):
 local=api.build('A',t0=t,n_factor=factors['n'],c_factor=factors['C']);local['T04_fixed_arrays']={k:getattr(local['model'],k).copy() for k in ['_n_ik','_Co']};return local

def evidence(local):
 m=local['model'];excess=m.h_Ik-local['raw_heights']
 assert all(np.array_equal(getattr(m,k),v) for k,v in local['T04_fixed_arrays'].items())
 return dict(final_time_s=float(m.t),finite=all(np.isfinite(getattr(m,k)).all() for k in api.STATE_KEYS),accepted_max_open_excess_m=float(excess.max()),accepted_open_exceeded=int((excess>1e-7).sum()),parameters_unchanged=True,values=api.sample(local),runtime_audit=copy.deepcopy(local['runtime_audit']))

def logged_call(local,phase,call):
 """只观测原step调用，不改变参数、步长、求解器返回或范围保护。"""
 m=local['model'];original=m.step;finished=Counter();failed=Counter();failures=[]
 def step(**kwargs):
  dt=float(kwargs['dt']);before=float(m.t)
  try:
   result=original(**kwargs);finished[str(dt)]+=1;return result
  except BaseException as ex:
   failed[str(dt)]+=1;failures.append(dict(time_s=before,dt_s=dt,error=repr(ex)));raise
 m.step=step
 try:return call(),dict(phase=phase,native_completed_calls_by_dt_s=dict(finished),native_failed_calls_by_dt_s=dict(failed),failed_native_calls=failures,note='预热native完成不等于adaptive接受；预热接受/拒绝以完整steady返回为准')
 finally:m.step=original

def certify(local,kw,label,prefix):
 (r,steps)=logged_call(local,label,lambda:api.steady(local,kw=kw,max_seconds=60.))
 state=evidence(local);good=bool(r['converged'] and state['finite'] and state['accepted_open_exceeded']==0)
 row=dict(label=label,certified=good,steady=r,state=state,step_audit=steps)
 dump(O/(prefix+'_'+label+'.json'),row)
 np.savez_compressed(O/(prefix+'_'+label+'_末态.npz'),**{k:getattr(local['model'],k) for k in api.STATE_KEYS},n=local['model']._n_ik,C=local['model']._Co)
 return row

results=[];started=time.time()
for wi,(lo,hi) in enumerate(windows,1):
 prefix=f'窗{wi}';stage='构模';local=None
 try:
  end=build(hi);mean=build(hi);history=build(hi-86400.)
  checks=[]
  for name,loc,kw in [('末时固定',end,api.forcing(end,hi)),('窗均值固定',mean,api.mean_forcing(mean,lo,hi)),('历史起点固定',history,api.forcing(history,hi-86400.))]:
   stage=name;local=loc;check=certify(loc,kw,name,prefix);checks.append(check)
   if not check['certified']:raise RuntimeError(name+'未获得原轮廓内认证稳态')
  stage='提前24h真实推进';local=history
  ((rows,retries),steps)=logged_call(history,stage,lambda:api.advance(history,hi-86400.,hi))
  df=pd.DataFrame(rows);assert len(df)==48 and df.time_s.iloc[-1]==hi
  df.to_csv(O/(prefix+'_提前24小时轨迹.csv'),index=False);last=df[df.time_s>=lo].copy();assert len(last)==13 and last.time_s.iloc[0]==lo
  last.to_csv(O/(prefix+'_最后六小时轨迹.csv'),index=False)
  actual_mean={k:float(np.trapezoid(last[k],last.time_s)/(hi-lo)) for k in api.sample(history)}
  tests=[]
  for name,x,y in [('末时固定vs到达末态',api.sample(end),api.sample(history)),('窗均值固定vs真实窗均值',api.sample(mean),actual_mean)]:
   head={o['name']:float(x[o['name']]-y[o['name']]) for o in mean['operators']};qdelta=float(x['gate_flow_m3s']-y['gate_flow_m3s']);qrel=abs(qdelta)/max(abs(y['gate_flow_m3s']),1.)
   tests.append(dict(mode=name,head_differences_m=head,head_pass={k:abs(v)<=.03 for k,v in head.items()},gate_difference_m3s=qdelta,gate_relative_difference=qrel,gate_pass=qrel<=.02,passed=all(abs(v)<=.03 for v in head.values()) and qrel<=.02,reference=x,actual=y))
  np.savez_compressed(O/(prefix+'_真实推进末态.npz'),**{k:getattr(history['model'],k) for k in api.STATE_KEYS},n=history['model']._n_ik,C=history['model']._Co)
  result=dict(window_index=wi,window=[lo,hi],status='PASSED' if all(t['passed'] for t in tests) else 'FAILED_T04_THRESHOLDS',steady_checks=checks,T04=tests,history_retries=retries,history_step_audit=steps,history_state=evidence(history))
 except BaseException as ex:
  result=dict(window_index=wi,window=[lo,hi],status='FAILED',stage=stage,error=repr(ex),traceback=traceback.format_exc(),state=evidence(local) if local is not None else None)
 results.append(result);dump(O/(prefix+'_终态结果.json'),result);print(json.dumps(dict(window=wi,status=result['status'],T04=result.get('T04'),step_audit=result.get('history_step_audit')),ensure_ascii=False),flush=True)
assert all(sha(p)==h for p,h in source_hashes.items())
result=dict(status='PASSED' if all(r['status']=='PASSED' for r in results) else 'FAILED_OR_PARTIAL',fixed_ratios=factors,windows=results,source_hashes_unchanged=True,runtime_identity=api.runtime_identity(),parameter_tuning=False,baseline_rerun=False,full_30day_rerun=False,wall_seconds=time.time()-started)
dump(O/'复核结果.json',result)
