"""不改数据，量化有效C对指定水位/流量假设的敏感性；不是仪器置信区间。"""
from pathlib import Path
import copy,json,sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

O=Path(__file__).resolve().parent;P=O.parent;R=P.parent
sys.path.insert(0,str(P/'03_局部模型'))
from 局部闸段 import build,mean_forcing,observed,steady,sample

definitions=[('原始条件',0.,0.,0.),('闸前目标高0.03m',.03,0.,0.),('闸前目标低0.03m',-.03,0.,0.),('尾水高0.03m',0.,.03,0.),('尾水低0.03m',0.,-.03,0.),('入口流量高2%',0.,0.,.02),('入口流量低2%',0.,0.,-.02)]
rows=[];trace=[]
for case in ['JZZ6_short','JZZ16_short']:
    cfg=json.loads((O/case/'运行约定.json').read_text());nominal=json.loads((O/case/'结果.json').read_text())
    lo,hi=cfg['train_window'];station=cfg['station'];upper=cfg['C_factor_bounds'][1]
    base=build(case,t0=hi);base_C=base['model']._Co.copy();mean=mean_forcing(base,lo,hi)
    for label,dy,dz,dq in definitions:
        target=cfg['target_time_weighted_mean_m']+dy
        kw=copy.deepcopy(mean);kw['H_bc'][base['audit']['cut_head_node']]+=dz;kw['Q_in'][base['audit']['cut_flow_node']]*=1+dq
        def error(factor):
            local=copy.deepcopy(base);local['model']._Co[:]=base_C*factor
            r=steady(local,kw=kw);assert r['converged']
            pred=sample(local)[station+'_forebay_m'];value=pred-target
            trace.append(dict(case=case,scenario=label,factor=float(factor),predicted_m=pred,target_m=target,residual_m=value,converged=True))
            return value
        low_error=error(1.);high_error=error(upper)
        if low_error*high_error<=0:
            factor=float(brentq(error,1.,upper,xtol=1e-7,rtol=1e-7));C=float(base_C[0]*factor);status='原参数界内可辨识'
            if label=='原始条件':assert abs(factor-nominal['factor'])<1e-7
        else:
            factor=None;C=None;status='沿增加C方向到上限仍无法匹配，未扩大原界'
        rows.append(dict(case=case,scenario=label,front_target_shift_m=dy,tail_shift_m=dz,flow_scale=1+dq,C_factor=factor,effective_C=C,upper_C=float(base_C[0]*upper),residual_at_upper_m=high_error,status=status))
pd.DataFrame(rows).to_csv(O/'有效系数对测量条件的敏感性.csv',index=False)
(O/'条件敏感性逐次记录.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))
(O/'条件敏感性说明.json').write_text(json.dumps(dict(scenarios=definitions,results=rows,note='各项是单独施加的诊断假设，不是仪表实测精度、统计置信区间或数据修正建议；仅固定窗口稳态辨识，不另行以这些假设计算后15天，也不据验证结果选情景。原始数据、名义辨识和系数界不改。'),ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
