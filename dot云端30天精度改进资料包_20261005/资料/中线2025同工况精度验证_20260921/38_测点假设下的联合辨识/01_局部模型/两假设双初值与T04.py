"""只做构模前提；不拟合。各假设独立，任一窗无认证稳态则停止该假设。"""
from pathlib import Path
import copy,hashlib,json,time,traceback
import numpy as np
import pandas as pd
import 测点假设局部模型 as api

O=Path(__file__).resolve().parent
WINDOWS=[(216000.,237600.),(648000.,669600.)]
dump=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=dict(windows=WINDOWS,head_tolerance_m=.03,relative_gate_flow_tolerance=.02,dual_guess_offset_m=.2,roughness_factor=1.,gate_coefficient_factor=1.,tailwater_not_scored=True,parameter_fitting=False,steady_acceptance='原adaptive认证且最终接受状态不超开敞原轮廓；被拒绝尝试单列',initialization='仅起算前合法低水面/流量猜测，不使用H16内部实测，不夹正式过程状态',api_hash=sha(api.__file__),harness_hash=sha(Path(__file__)))
dump(O/'前提核验约定.json',contract)

def state_evidence(local):
 m=local['model'];finite=all(np.isfinite(getattr(m,k)).all() for k in api.STATE_KEYS);excess=m.h_Ik-local['raw_heights'];closed=np.isinf(local['raw_heights'])
 return dict(finite=finite,accepted_max_open_excess_m=float(excess.max()),accepted_open_exceeded=int((excess>1e-7).sum()),closed_pressure_nodes=int(closed.sum()),values=api.sample(local),attempt_audit=copy.deepcopy(local['runtime_audit']))

def certify(local,kw,label,target):
 result=api.steady(local,kw=kw,max_seconds=60.);evidence=state_evidence(local)
 valid=bool(result['converged'] and evidence['finite'] and evidence['accepted_open_exceeded']==0)
 row=dict(label=label,certified=valid,steady=result,state=evidence)
 dump(target,row);return row

summary=[]
for hyp in ['A','B']:
 completed=[];stopped=None;started=time.time()
 try:
  for wi,(lo,hi) in enumerate(WINDOWS,1):
   prefix=f'{hyp}_窗{wi}'
   mean=api.build(hyp,t0=hi);high=api.build(hyp,t0=hi,initial_offset=.2);end=api.build(hyp,t0=hi)
   dump(O/(prefix+'_低初值构模.json'),mean['audit']);dump(O/(prefix+'_高初值构模.json'),high['audit']);hold=api.mean_forcing(mean,lo,hi)
   checks=[]
   for label,local,kw in [('窗均值低初值',mean,hold),('窗均值高初值',high,hold),('末时固定',end,api.forcing(end,hi))]:
    row=certify(local,kw,label,O/(prefix+'_'+label+'.json'));checks.append(row)
    if not row['certified']:stopped=dict(window_index=wi,window=[lo,hi],stage=label,reason='未获得原轮廓内认证稳态',evidence=row);break
   if stopped:break
   full_delta={k:float(np.max(abs(getattr(mean['model'],k)-getattr(high['model'],k)))) if getattr(mean['model'],k).size else 0. for k in api.STATE_KEYS}
   target_delta={o['name']:abs(api.sample(mean)[o['name']]-api.sample(high)[o['name']]) for o in mean['operators']}
   if max(target_delta.values())>1e-5:stopped=dict(window_index=wi,stage='双初值一致性',target_differences=target_delta,full_state_difference=full_delta);break
   hist=api.build(hyp,t0=hi-86400.);hist_init=certify(hist,api.forcing(hist,hi-86400.),'提前24小时起点',O/(prefix+'_历史起点稳态.json'))
   if not hist_init['certified']:stopped=dict(window_index=wi,stage='历史起点未认证',evidence=hist_init);break
   rows,retries=api.advance(hist,hi-86400.,hi);df=pd.DataFrame(rows);df.to_csv(O/(prefix+'_提前24小时轨迹.csv'),index=False);w=df[df.time_s>=lo];assert len(w)==13
   mean_actual={k:float(np.trapezoid(w[k],w.time_s)/(hi-lo)) for k in api.sample(hist)}
   t04=[]
   for mode,x,y in [('末时固定vs到达末态',api.sample(end),api.sample(hist)),('窗均值固定vs真实窗均值',api.sample(mean),mean_actual)]:
    diff={o['name']:float(x[o['name']]-y[o['name']]) for o in mean['operators']};qd=float(abs(x['gate_flow_m3s']-y['gate_flow_m3s'])/max(abs(y['gate_flow_m3s']),1.))
    t04.append(dict(mode=mode,head_differences_m=diff,relative_gate_flow_difference=qd,passed=max(abs(v) for v in diff.values())<=.03 and qd<=.02,reference=x,actual=y))
   row=dict(window_index=wi,window=[lo,hi],steady_checks=checks,dual_initial_state_difference=full_delta,dual_target_difference=target_delta,T04=t04,history_retries=retries,history_state=state_evidence(hist),eligible=all(q['passed'] for q in t04))
   completed.append(row);dump(O/(prefix+'_前提结果.json'),row);print(json.dumps(dict(hypothesis=hyp,window=wi,T04=t04),ensure_ascii=False),flush=True)
   if not row['eligible']:stopped=dict(window_index=wi,stage='T04差值门槛',evidence=t04);break
 except BaseException as ex:
  local_for_evidence=hist if 'hist' in locals() and hist.get('hypothesis')==hyp else mean
  stopped=dict(reason=repr(ex),traceback=traceback.format_exc(),state=state_evidence(local_for_evidence))
 result=dict(hypothesis=hyp,status='PREREQUISITES_PASSED' if not stopped and len(completed)==2 else 'STOPPED_AT_PREDECLARED_GATE',eligible_for_conditional_joint_trial=not stopped and len(completed)==2,completed_windows=completed,stopped=stopped,runtime_identity=api.runtime_identity(),api_unchanged=sha(api.__file__)==contract['api_hash'],wall_s=time.time()-started,not_physical_truth=True)
 assert result['api_unchanged'];dump(O/(hyp+'_前提总结果.json'),result);summary.append(result);print(json.dumps(dict(hypothesis=hyp,status=result['status'],completed_windows=len(completed),wall_s=result['wall_s']),ensure_ascii=False),flush=True)
dump(O/'两假设前提总结果.json',summary)
