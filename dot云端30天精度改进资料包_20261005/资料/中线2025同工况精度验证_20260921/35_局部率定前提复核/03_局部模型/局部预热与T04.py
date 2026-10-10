from pathlib import Path
import json
import numpy as np
import pandas as pd
from 局部闸段 import build,forcing,mean_forcing,sample,steady,advance,STATE_KEYS

O=Path(__file__).resolve().parent
criteria={'head_difference_m':.03,'gate_flow_relative_difference':.02,'mean_method':'trapezoidal time weighting','window_selection':'沿用30目录目标闸预设门槛候选，不按模型误差选窗','scope':'短段模型内前提；尾水是边界不评分；不开展C辨识；JZZ17桩号疑点使长段限构模'}
(O/'T04预定判据.json').write_text(json.dumps(criteria,ensure_ascii=False,indent=2))

def physical(local):
 m=local['model'];heights=np.zeros(m.NIk)
 for meta in local['converted']['channel_discretization'].values():
  for p in meta['sections']:heights[p['junction_id']]=p['height']
 return dict(state_finite=all(np.isfinite(getattr(m,k)).all() for k in STATE_KEYS),raw_contour_exceeded=int((m.h_Ik>heights+1e-8).sum()),minimum_contour_margin_m=float((heights-m.h_Ik).min()),minimum_gate_Q=float(m.Q_o.min()))

rows=[]
for case,lo,hi in [('JZZ6_short',153000.,174600.),('JZZ16_short',428400.,450000.)]:
 end=build(case,t0=hi);mean=build(case,t0=hi);high=build(case,t0=hi,initial_offset=.2)
 kw=mean_forcing(mean,lo,hi)
 states=[]
 for name,loc,hold in [('末时固定',end,forcing(end,hi)),('窗均值固定低初值',mean,kw),('窗均值固定高初值',high,kw)]:
  result=steady(loc,kw=hold);states.append(dict(mode=name,steady=result,values=sample(loc),physical=physical(loc)))
  (O/(case+'_'+name+'_稳态.json')).write_text(json.dumps(states[-1],ensure_ascii=False,indent=2))
  if not result['converged']:raise RuntimeError((case,name,result['status']))
 diff={k:abs(sample(mean)[k]-sample(high)[k]) for k in sample(mean)}
 history=build(case,t0=hi-86400.);init=steady(history,t=hi-86400.)
 if not init['converged']:raise RuntimeError((case,'historical_start',init['status']))
 trajectory,retries=advance(history,hi-86400.,hi);frame=pd.DataFrame(trajectory);frame.to_csv(O/(case+'_提前24小时真实推进.csv'),index=False)
 w=frame[frame.time_s>=lo];assert len(w)==13
 mean_actual={k:float(np.trapezoid(w[k],w.time_s)/(hi-lo)) for k in sample(history)}
 results=[]
 for mode,reference,actual in [('末时固定vs到达末态',sample(end),sample(history)),('窗均值固定vs真实窗均值',sample(mean),mean_actual)]:
  errors={o['name']:float(reference[o['name']]-actual[o['name']]) for o in history['operators']}
  qerr=float(abs(reference['gate_flow_m3s']-actual['gate_flow_m3s'])/max(abs(actual['gate_flow_m3s']),1.))
  results.append(dict(mode=mode,internal_head_differences_m=errors,gate_flow_relative_difference=qerr,head_pass=all(abs(v)<=.03 for v in errors.values()),flow_pass=qerr<=.02,passed=all(abs(v)<=.03 for v in errors.values()) and qerr<=.02,reference=reference,actual=actual))
 row=dict(case=case,window_start=lo,window_end=hi,steady_states=states,dual_initial_difference=diff,history_initial_steady=init,history_end_physical=physical(history),history_retries=retries,T04=results)
 rows.append(row);print(json.dumps(dict(case=case,dual_initial_difference=diff,T04=results),ensure_ascii=False),flush=True)
(O/'短段双初值与T04结果.json').write_text(json.dumps(dict(criteria=criteria,results=rows),ensure_ascii=False,indent=2))
