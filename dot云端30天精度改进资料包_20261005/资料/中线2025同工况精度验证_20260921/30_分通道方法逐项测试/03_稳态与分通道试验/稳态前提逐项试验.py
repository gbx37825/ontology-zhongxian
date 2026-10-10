from pathlib import Path
import json,time,copy,numpy as np,pandas as pd
from 试验水力工具 import *
O=Path(__file__).resolve().parent;target=790200.;past=target-86400.;start=time.time();result={};ops=None
for label,t,shift in [('目标时刻恒边界稳态',target,0),('不同初值恒边界稳态',target,.3),('提前24小时恒边界稳态',past,0)]:
 e,s,c,f,params=build();m=e.model
 if shift:
  state={k:getattr(m,k).copy() for k in STATE_KEYS};state['H_j'][~m.bc]+=shift;state['h_Ik']+=shift;restore(m,state)
 ops=operators(e,s,c,['JZZ4','JZZ6','JZZ15','JZZ16','JZZ17']);diag=fixed_steady(e,f,t,O,label)
 result[label]=dict(converged=diag['converged'],status=diag['status'],sim_time=t,initial_head_shift_m=shift,observations=values(m,ops))
 if label=='目标时刻恒边界稳态':base=copy.deepcopy(e);base_f=f;base_ops=ops;converted=c;original_source=s;parameter_layout=params
 if label=='提前24小时恒边界稳态' and diag['converged']:
  m.t=past;m.iter_count=0;stats=advance(m,f,past,target);save(m,O/'提前24小时真实输入到达状态.npz');result['提前24小时真实输入到达']=dict(completed=True,observations=values(m,ops),retries=stats[0]);print('提前24h路径完成',flush=True)
 (O/'逐项进度.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
# Compare independent target paths, not just printed RUNNING/convergence labels.
if result['目标时刻恒边界稳态']['converged']:
 v0=result['目标时刻恒边界稳态']['observations'];comp=[]
 for label in ['不同初值恒边界稳态','提前24小时真实输入到达']:
  if label not in result or not result[label].get('converged',result[label].get('completed',False)):continue
  for name,value in result[label]['observations'].items():comp.append(dict(comparison=label,operator=name,base=v0[name],other=value,difference=value-v0[name]))
 pd.DataFrame(comp).to_csv(O/'双初值与提前起算比较.csv',index=False)
 # Freeze external input and perturb one physically grouped parameter at a time.
 sensitivity=[]
 for par in ['POOL15_n','POOL16_n','JZZ16_C']:
  for factor in [.95,1.05]:
   e=copy.deepcopy(base);change(e.model,parameter_layout,par,factor);label=par+('_减5' if factor<1 else '_加5');d=fixed_steady(e,base_f,target,O,label)
   if d['converged']:
    for name,v in values(e.model,base_ops).items():sensitivity.append(dict(parameter=par,factor=factor,operator=name,delta=v-v0[name],base=v0[name],changed=v))
   result[label]=dict(converged=d['converged'],status=d['status'])
   (O/'逐项进度.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 pd.DataFrame(sensitivity).to_csv(O/'稳态单参数灵敏度.csv',index=False)
result['wall_seconds']=time.time()-start;result['measured_steady_calibration_eligible']=False;result['note']='全线实测窗口未满足预注册准稳态门槛；上述是固定输入的模型内试验，不据此把稳态值当实测真值。'
(O/'稳态前提试验汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('FINISHED',json.dumps(result,ensure_ascii=False),flush=True)
