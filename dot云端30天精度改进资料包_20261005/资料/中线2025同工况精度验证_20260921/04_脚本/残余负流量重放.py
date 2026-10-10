"""同C03输入/同失稳前初态，复现并单独替换三站闸流关系；不改实测输入。"""
from pathlib import Path
import json,argparse,time,types
import numpy as np,pandas as pd
from 试验公共 import R
from 闸流振荡根因核验 import build
O=R/'15_残余负流量根因';O.mkdir(exist_ok=True);p=argparse.ArgumentParser();p.add_argument('--case',choices=['original','integral_only_tangent'],required=True);p.add_argument('--iterations',type=int,default=1);p.add_argument('--start',type=int,default=1996200);p.add_argument('--end',type=int,default=2592000);a=p.parse_args()
e,s,c,f=build(a.case,'Q03_六点开度修补同工况30天',a.start,str(R/'14_输入修补交付/C03_含七里河重点脉冲修补候选_2025同工况30天.json'),'legacy');m=e.model
if a.case!='original':m._only=c['orifices'].name.str.match(r'^JZZ(21|23|27)-').to_numpy()
groups={st:np.flatnonzero(c['orifices'].name.str.startswith(st+'-')) for st in ['JZZ21','JZZ23','JZZ27']};frames=[];first={};negative_steps={st:0 for st in groups};minimum={st:float('inf') for st in groups};failure_steps=0;maxres=0;startwall=time.time();badrows=[]
for i,t in enumerate(range(a.start,a.end,15)):
 kw=f.at(t);kw['num_iter']=a.iterations;before=m.H_j.copy();qbefore=m.Q_o.copy();m._heads=[];m.step(**kw);last=m._heads[-2] if len(m._heads)>1 else before;res=float(abs(m.H_j-last).max());maxres=max(maxres,res);failure_steps+=int(a.iterations>1 and res>=.0015)
 lin=m._alpha_o*m.H_j[m._J_uo]+m._beta_o*m.H_j[m._J_do]+m._chi_o
 for st,ids in groups.items():
  up=m._J_uo[ids[0]];dn=m._J_do[ids[0]];q=float(m.Q_o[ids].sum());minimum[st]=min(minimum[st],q)
  row=dict(station=st,time_s=t+15,hu=float(m.H_j[up]),hd=float(m.H_j[dn]),q=q,head_residual=res,linear_flow_mismatch=float(abs(lin[ids]-m.Q_o[ids]).max()))
  if (t+15)%1800==0:frames.append(row)
  if q<0:
   negative_steps[st]+=1;badrows.append(row)
   if st not in first:
    hs=[]
    for k in ids:
     zb=m._z_inv_j[m._J_uo[k]]+m._z_o[k];opening=kw['u_o'][k]*m._y_max_o[k];hi=max(before[up],before[dn])-zb
     hs.append(dict(name=str(c['orifices'].iloc[k]['name']),opening=float(opening),pre_hi_depth=float(hi),pre_weir_branch=bool(opening>=hi and hi>0),prior_q=float(qbefore[k]),final_q=float(m.Q_o[k]),linear_q=float(lin[k])))
    first[st]=dict(**row,pre_hu=float(before[up]),pre_hd=float(before[dn]),holes=hs)
 if not np.isfinite(m.H_j).all():raise ValueError('nonfinite')
 if (i+1)%5760==0:print(a.case,'simulated_days',(i+1)/5760,flush=True)
name=a.case+'_iter'+str(a.iterations);pd.DataFrame(frames).to_csv(O/(name+'_30分钟输出.csv'),index=False);pd.DataFrame(badrows).to_csv(O/(name+'_15秒负流量.csv'),index=False)
report=dict(case=a.case,iterations=a.iterations,start=a.start,end=a.end,negative_steps=negative_steps,min_q=minimum,first_negative=first,unconverged_steps=failure_steps if a.iterations>1 else None,max_head_iter_residual=maxres if a.iterations>1 else None,wall_seconds=time.time()-startwall,steps=i+1,finite=True)
if a.case=='original' and a.iterations==1:
 original=pd.read_csv(R/'06_结果/Q03_六点开度修补同工况30天/逐闸水位流量.csv');v=pd.DataFrame(frames).merge(original,left_on=['station','time_s'],right_on=['station_code','time_s']);report['original_replay_max_difference']=max(float(abs(v.hu-v.forebay_m).max()),float(abs(v.hd-v.afterbay_m).max()),float(abs(v.q-v.flow_m3s).max()));assert report['original_replay_max_difference']<1e-8
(O/(name+'_摘要.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
