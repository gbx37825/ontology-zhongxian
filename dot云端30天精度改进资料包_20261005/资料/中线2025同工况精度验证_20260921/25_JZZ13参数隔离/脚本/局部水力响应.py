from pathlib import Path
import sys,json,copy,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'25_JZZ13参数隔离';P=R/'24_参数同化根因核查'
sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
from 试验公共 import Forcing
from 并联结构物理修复 import build_engine_physics
from 常驻水力进程 import advance
from 局地参数EnKF import STATE_KEYS,restore
source=P/'F01_拓扑分组修复六小时';cfg=json.loads((source/'运行配置.json').read_text());e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));m=e.model;forcing=Forcing(e,s,c)
with np.load(source/'末态集合.npz') as z:
 restore(m,{k:z[f'm0_{k}'].copy() for k in STATE_KEYS});theta=z['parameter_log_ratios'][:,0]
m.t=21600;m.iter_count=0
for j,p in enumerate(cfg['parameter_bounds']):
 arr=m._n_ik if p['kind']=='n' else m._Co;arr[p['indices']]*=np.exp(theta[j])
m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk]
groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1)) for st,g in c['orifices'].assign(station=c['orifices'].name.str.split('-').str[0]).groupby('station')};rows=[]
for case,parameter,factor in [('基准',None,1),('糙率减10%','POOL12_n',.9),('糙率加10%','POOL12_n',1.1),('闸流系数减10%','JZZ13_C',.9),('闸流系数加10%','JZZ13_C',1.1)]:
 mm=copy.deepcopy(m)
 if parameter:
  p=next(p for p in cfg['parameter_bounds'] if p['name']==parameter);arr=mm._n_ik if p['kind']=='n' else mm._Co;arr[p['indices']]*=factor;mm._n_uk[:]=mm._n_ik[mm._i_1k];mm._n_dk[:]=mm._n_ik[mm._i_nk]
 for end in [25200,32400,43200]:
  advance(mm,forcing,mm.t,end)
  for st in ['JZZ12','JZZ13','JZZ14']:
   for met,j in zip(['forebay_m','afterbay_m'],groups[st]):rows.append(dict(case=case,time_s=end,station=st,metric=met,level=float(mm.H_j[j])))
a=pd.DataFrame(rows);b=a[a.case=='基准'].drop(columns='case').rename(columns={'level':'baseline_level'});a=a.merge(b,on=['time_s','station','metric']);a['delta_m']=a.level-a.baseline_level;a.to_csv(O/'同断点局部参数水力响应.csv',index=False);print(a[(a.time_s==43200)&(a.station=='JZZ13')].to_string(index=False))
