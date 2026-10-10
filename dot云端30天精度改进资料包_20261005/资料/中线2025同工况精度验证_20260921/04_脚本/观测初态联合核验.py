"""联合初态的水位-only控制身份、首步及连续方程核验。只读运行结果。"""
from pathlib import Path
import ast,json,copy,types,shutil
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from 试验公共 import *
from 观测初态联合运行 import snap,restore,flux_residual
r=R/'06_结果';small=r/'IQ00_作者88侧水位与流量初态6小时';long=r/'IQ01_作者88侧水位与流量初态6天'
norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());e,source,c=build_engine(gate_max_normalization=norm);f=Forcing(e,source,c,positioned=True);warmup(e,source,f);m=e.model
measured=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');init=pd.read_csv(R/'02_输入/仅作者88侧初始化观测.csv');obs=init[init.metric!='flow_m3s'].set_index(['station_code','metric','time_s'])
g=c['orifices'].copy();g['station']=g.name.str.extract(r'^(JZZ\d+)-')[0];groups={s:(int(gg.iloc[0].sj_0),int(gg.iloc[0].sj_1),gg.id.to_numpy(int)) for s,gg in g.groupby('station')}
history=[{(s,k):float(m.H_j[u if k=='forebay_m' else d]) for s,(u,d,gi) in groups.items() for k in ['forebay_m','afterbay_m']}]
# 原封不动执行主runner内的initialize函数体，避免手工复写控制算法。
module=ast.parse((R/'04_脚本/运行水力对照.py').read_text());node=next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name=='initialize')
env=globals().copy();env.update(m=m,e=e,groups=groups,obs=obs,history=history,history_times=[0.],updates=[],a=types.SimpleNamespace(initialize='profile',init_lookback_hours=6.,init_cap=2.,name='IQ主runner初态控制核验'))
exec(compile(ast.Module(body=[node],type_ignores=[]),'原主runner_initialize','exec'),env);env['initialize'](0)
with np.load(long/'初态_0h.npz') as data:
    candidate={k:data[k] for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
head_diff={k:float(abs(getattr(m,k)-candidate[k]).max()) for k in ['H_j','h_Ik']};assert max(head_diff.values())==0,head_diff
restore(m,candidate);kw=f.at(0);residual=flux_residual(m,kw);dynamic=np.r_[~m.bc,np.ones(m.NIk,dtype=bool)]
m.node_coeffs(_Q_0Ik=kw['Q_0Ik'],_dt=15);area=np.r_[m._A_sj,m._E_Ik*15];rate=np.divide(residual,area,out=np.full_like(area,np.nan),where=area>1e-10);rate[~dynamic]=np.nan
nodal=pd.DataFrame(dict(kind=['superjunction']*m.M+['internal_junction']*m.NIk,index=list(range(m.M))+list(range(m.NIk)),fixed_head_boundary=~dynamic,required_storage_rate_m3s=residual,effective_surface_area_m2=area,implied_level_rate_m_per_h=rate*3600));nodal.to_csv(R/'07_对比分析/IQ初态连续方程诊断.csv',index=False)
base=snap(m);m.step(**kw);j=int(np.argmax(abs(m.H_j-base['H_j'])));q=int(np.argmax(abs(m.Q_ik-base['Q_ik'])))
summary=dict(parent_head_only_initial_max_differences=head_diff,dynamic_required_storage_abs_sum_m3s=float(abs(residual[dynamic]).sum()),dynamic_required_storage_max_m3s=float(abs(residual[dynamic]).max()),dynamic_storage_net_m3s=float(residual[dynamic].sum()),zero_storage_nonzero_residual_count=int(((area<1e-10)&(abs(residual)>1e-6)&dynamic).sum()),fixed_head_boundary_terms=[{**row,'interpretation':'固定水位边界的库外交换通量，不计为模型蓄量变化或水位变化率'} for row in nodal[nodal.fixed_head_boundary].drop(columns=['implied_level_rate_m_per_h']).to_dict('records')],first_step_max_head_node=e.node_names[j],first_step_head_change_m=float(m.H_j[j]-base['H_j'][j]),first_step_max_internal_flow_link=q,first_step_max_internal_flow_channel=e.channel_names[int(m._ki[q])],first_step_internal_flow_change_m3s=float(m.Q_ik[q]-base['Q_ik'][q]))
a=pd.read_csv(small/'逐闸水位流量.csv');b=pd.read_csv(long/'逐闸水位流量.csv');join=a.merge(b[b.time_s<=21600],on=['station_code','time_s'],suffixes=('_small','_long'),validate='one_to_one');summary['six_hour_prefix_max_difference']=max(float(abs(join[x+'_small']-join[x+'_long']).max()) for x in ['forebay_m','afterbay_m','flow_m3s']);assert summary['six_hour_prefix_max_difference']==0
(R/'07_对比分析/IQ联合初态核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
rows=[]
for name in ['B00_实际开度同工况30天','I00_观测水面线起步30天','I01_作者88侧水面线起步30天','IQ01_作者88侧水位与流量初态6天']:
    frame=pd.read_csv(r/name/'逐闸水位流量.csv').melt(id_vars=['station_code','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(measured,on=['station_code','time_s','metric'],validate='one_to_one');frame=frame[frame.valid];frame['error']=frame.sim-frame.value
    for h in [6,24,72,144]:
        for scope in ['全118侧','作者88侧']:
            d=frame[(frame.time_s==h*3600)&(frame.pdf_comparable if scope=='作者88侧' else True)];rows.append(dict(run=name,hour=h,scope=scope,samples=len(d),MAE=d.error.abs().mean(),max_abs=d.error.abs().max(),bias=d.error.mean()))
    for scope in ['全118侧','作者88侧']:
        d=frame[(frame.time_s>0)&(frame.time_s<=518400)&(frame.pdf_comparable if scope=='作者88侧' else True)];rows.append(dict(run=name,hour='0至144h不含初值',scope=scope,samples=len(d),MAE=d.error.abs().mean(),max_abs=d.error.abs().max(),bias=d.error.mean()))
pd.DataFrame(rows).to_csv(R/'07_对比分析/IQ与水位单独初始化对照.csv',index=False)
for log,folder in [('/private/tmp/IQ00_6h.log',small),('/private/tmp/IQ01_6d.log',long)]:shutil.copy2(log,folder/'运行日志.txt')
