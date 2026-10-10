"""只验证分通道辨识实现；目标水位来自已知参数模型，不是实测精度结果。"""
from pathlib import Path
import time,json,copy,numpy as np,pandas as pd
from 试验水力工具 import *
from pipedream_solver.gate_hydraulics import rectangular_gate_discharge
O=Path(__file__).resolve().parent/'合成验证';O.mkdir(exist_ok=True);target=790200.;e,s,c,f,params=build();m=e.model
parent=O.parent
base_diag=json.loads((parent/'目标时刻恒边界稳态_稳态诊断.json').read_text());assert base_diag['converged']
with np.load(parent/'目标时刻恒边界稳态_状态.npz') as z:restore(m,{k:z[k] for k in STATE_KEYS})
ops=operators(e,s,c,['JZZ15','JZZ16','JZZ17']);base=copy.deepcopy(e);known={'POOL16_n':1.08,'JZZ16_C':.94};result={'kind':'合成已知参数回收，非实测率定','known_ratios':known,'iterations':[]};started=time.time()
for name,ratio in known.items():change(m,params,name,ratio)
d=fixed_steady(e,f,target,O,'合成真值')
if not d['converged']:raise RuntimeError('合成真值稳态未通过，禁止继续')
true=copy.deepcopy(e);truth=values(m,ops);gateids=c['orifices'].index[c['orifices'].name.str.startswith('JZZ16-')].to_numpy();measured_q=float(m.Q_o[gateids].sum());unit=0.
for i in gateids:
 row=c['orifices'].iloc[i];ju,jd=int(row.sj_0),int(row.sj_1);z=m._z_inv_j[ju]+float(row.z_o)
 unit+=rectangular_gate_discharge(float(m.H_j[ju]-z),float(m.H_j[jd]-z),float(row.y_max*f.at(target)['u_o'][i]),float(row.g2),float(base.model._Co[i]))
C_ratio=measured_q/unit;result['recovered_C_ratio']=C_ratio;assert abs(C_ratio-.94)<1e-4
# With C fixed, solve one n using a channel-water-level observation and finite-difference response.
ratio=1.;obsname='JZZ16_afterbay_m';work=copy.deepcopy(base);change(work.model,params,'JZZ16_C',C_ratio)
for iteration in range(4):
 pred=copy.deepcopy(work);change(pred.model,params,'POOL16_n',ratio);diag=fixed_steady(pred,f,target,O,f'n迭代{iteration}')
 if not diag['converged']:raise RuntimeError('n回代稳态未通过')
 value=values(pred.model,ops)[obsname];error=truth[obsname]-value
 row=dict(iteration=iteration,n_ratio=ratio,target=truth[obsname],predicted=value,residual_m=error);result['iterations'].append(row)
 if abs(error)<=.001:break
 probe=copy.deepcopy(pred);change(probe.model,params,'POOL16_n',1.02);dia=fixed_steady(probe,f,target,O,f'n增益{iteration}')
 if not dia['converged']:raise RuntimeError('n增益稳态未通过')
 sensitivity=(values(probe.model,ops)[obsname]-value)/np.log(1.02);row['sensitivity_m_per_log_n']=sensitivity
 if abs(sensitivity)<.01:raise RuntimeError('糙率灵敏度不足，禁止反演')
 ratio=float(np.clip(ratio*np.exp(np.clip(error/sensitivity,-np.log(1.05),np.log(1.05))),.8,1.2))
result['recovered_n_ratio']=row['n_ratio'];result['water_fit_residual_m']=row['residual_m'];result['n_relative_error']=abs(row['n_ratio']/1.08-1);result['calibration_pass']=bool(abs(row['residual_m'])<=.001 and result['n_relative_error']<=.005)
# A different, time-varying forcing segment; no observations or parameter updates.
rows=[];fit=pred
for engine,label in [(true,'合成真值参数'),(fit,'回收参数')]:
 engine.model.t=target;engine.model.iter_count=0
 for k in range(12):
  advance(engine.model,f,target+k*1800,target+(k+1)*1800)
  rows.append(dict(run=label,time_s=target+(k+1)*1800,**values(engine.model,ops)))
df=pd.DataFrame(rows);df.to_csv(O/'异边界六小时合成检验.csv',index=False);a=df[df.run=='合成真值参数'].drop(columns=['run','time_s']).to_numpy();b=df[df.run=='回收参数'].drop(columns=['run','time_s']).to_numpy();result['holdout_max_difference_each_operator']=dict(zip(df.columns[2:],np.max(abs(a-b),axis=0).tolist()));result['wall_seconds']=time.time()-started
(O/'合成回收核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
