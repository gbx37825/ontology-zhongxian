"""从同一物理模型4h状态起步，核验JZZ42调闸异常对时间步长的敏感性。"""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
from 试验公共 import R,Forcing
from 并联结构物理修复 import build_engine_physics
p=argparse.ArgumentParser();p.add_argument('--base-run',default='P00_并联物理修复稳态6小时');p.add_argument('--out-name',default='PHV00_并联物理调闸步长窗');p.add_argument('--model',default=str(R/'02_输入/C00_实际开度负值修复_2025同工况30天.json'));p.add_argument('--audit-name',default='并联物理步长敏感性.json');args=p.parse_args()
name=args.base_run;base=R/'06_结果'/name;out=R/'06_结果'/args.out_name;out.mkdir(exist_ok=False)
with np.load(base/'内部断面状态.npz') as d:
    j=int(np.flatnonzero(d['time_s']==14400)[0]);initial={k:d[k][j].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());results=[]
for dt in [15.,7.5,3.75]:
    e,source,c=build_engine_physics(args.model,gate_max_normalization=norm);m=e.model;f=Forcing(e,source,c,positioned=True);m.load_state({**initial,'t':14400.})
    g=c['orifices'];gg=g[g.name.str.startswith('JZZ42-')];up=int(gg.iloc[0].sj_0);down=int(gg.iloc[0].sj_1);ids=gg.id.to_numpy(int);rows=[];max_dh=0.
    for i in range(int(7200/dt)):
        t=14400+i*dt;kw=f.at(t);kw['dt']=dt;before=m.H_j.copy();m.step(**kw);max_dh=max(max_dh,float(abs(m.H_j-before).max()));now=t+dt
        if not np.isfinite(m.H_j).all() or not np.isfinite(m.Q_ik).all():raise FloatingPointError((dt,now))
        rows.append(dict(time_s=now,forebay_m=m.H_j[up],afterbay_m=m.H_j[down],gate_flow=m.Q_o[ids].sum()))
    df=pd.DataFrame(rows);df.to_csv(out/f'dt_{dt:g}s_JZZ42.csv',index=False)
    if dt==15:
        original=pd.read_csv(base/'逐闸水位流量.csv');original=original[(original.station_code=='JZZ42')&(original.time_s>14400)];join=df.merge(original,on='time_s',suffixes=('_window','_original'));maxdiff=max(float(abs(join[k+'_window']-join[k+'_original']).max()) for k in ['forebay_m','afterbay_m']);maxdiff=max(maxdiff,float(abs(join.gate_flow-join.flow_m3s).max()));assert maxdiff<1e-10,maxdiff
    results.append(dict(dt=dt,min_forebay=float(df.forebay_m.min()),max_forebay=float(df.forebay_m.max()),min_gate_flow=float(df.gate_flow.min()),max_gate_flow=float(df.gate_flow.max()),max_all_node_step_change=max_dh,point_5p5h=df[df.time_s==19800].to_dict('records')[0]))
    print(results[-1],flush=True)
a=pd.read_csv(out/'dt_7.5s_JZZ42.csv');b=pd.read_csv(out/'dt_3.75s_JZZ42.csv');match=a.merge(b,on='time_s',suffixes=('_7p5','_3p75'));diff={k:float(abs(match[k+'_7p5']-match[k+'_3p75']).max()) for k in ['forebay_m','afterbay_m','gate_flow']}
result=dict(status='COMPLETED',base_run=name,model=args.model,start_h=4,end_h=6,initial_state='same 15s physical model state at4h',same_dt_restart_max_difference=maxdiff,dt_cases=results,dt_7p5_vs_3p75_max_differences=diff)
(out/'核验摘要.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(R/'07_对比分析'/args.audit_name).write_text(json.dumps(result,ensure_ascii=False,indent=2))
