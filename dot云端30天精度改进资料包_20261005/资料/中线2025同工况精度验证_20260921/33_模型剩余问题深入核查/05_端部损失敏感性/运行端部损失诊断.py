from pathlib import Path
import sys,json,time,hashlib,argparse,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];B=R/'26_冻结基线与后续归因';sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
from 试验公共 import Forcing
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import STATE_KEYS,restore
from 常驻水力进程 import advance
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--factor',type=float,required=True);a=p.parse_args();O=Path(__file__).resolve().parent/a.name;O.mkdir(exist_ok=False);beg=time.time();e,s,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()));m=e.model;f=Forcing(e,s,c);src=B/'B02_渠池归属修复后的参数同化诊断基线/六小时诊断起点';cfg=json.loads((src/'运行配置.json').read_text())
with np.load(src/'末态集合.npz') as z:
 restore(m,{k:z[f'm0_{k}'].copy() for k in STATE_KEYS});theta=z['parameter_log_ratios'][:,0]
for i,p in enumerate(cfg['parameter_bounds']):
 arr=m._n_ik if p['kind']=='n' else m._Co;arr[p['indices']]*=np.exp(theta[i])
m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m.t=21600.;m.iter_count=0
ks=c['superlinks'].index[c['superlinks'].name.str.upper().str.startswith('POOL')].to_numpy();original_up=m._C_uk.copy();original_down=m._C_dk.copy();m._C_uk[ks]*=a.factor;m._C_dk[ks]*=a.factor;assert len(ks)==126
config=dict(name=a.name,member=0,start_s=21600,end_s=259200,simulation='无同化单因素诊断',factor=a.factor,changed_superlinks=len(ks),changed_end_items=2*len(ks),initial_K=float(original_up[ks[0]]),applied_K=float(m._C_uk[ks[0]]),source_checkpoint_sha256=hashlib.sha256((src/'末态集合.npz').read_bytes()).hexdigest(),notes='只缩放126条开敞渠道端头当前Numba局损因子，不动105倒虹吸源K，不动糙率或外部节制闸流量系数。0.8/1.2不是有工程来源的新参数，只测试敏感性。')
(O/'运行配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};rows=[];retries=0
try:
 for k in range(132):
  t=21600+k*1800;stats=advance(m,f,t,t+1800);retries+=stats[0]
  for st,(ju,jd) in groups.items():rows.append(dict(time_s=t+1800,station=st,forebay_m=m.H_j[ju],afterbay_m=m.H_j[jd],gate_flow_m3s=float(m.Q_o[[i for name,i in e.gate_index.items() if name.startswith(st+'-')]].sum())))
  if (k+1)%48==0:print('periods',k+1,flush=True)
 pd.DataFrame(rows).to_csv(O/'水位流量.csv',index=False);np.savez_compressed(O/'末态.npz',**{k:getattr(m,k) for k in STATE_KEYS});result=dict(status='COMPLETED',periods=132,retries=retries,wall_seconds=time.time()-beg)
except Exception as ex:
 result=dict(status='FAILED',error=repr(ex),periods=len(rows)/59,wall_seconds=time.time()-beg);raise
finally:(O/'结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
