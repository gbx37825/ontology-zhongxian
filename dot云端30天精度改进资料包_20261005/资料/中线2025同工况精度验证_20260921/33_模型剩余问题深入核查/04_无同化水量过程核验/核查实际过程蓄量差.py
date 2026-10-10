"""以B02共同6小时检查点成员0无同化推进，核对开敞内部连接两种蓄量表达，不修改求解器。"""
from pathlib import Path
import sys,json,time,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;B=R/'26_冻结基线与后续归因';sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
from 试验公共 import Forcing
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import STATE_KEYS,restore
start=time.time();e,s,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()));m=e.model;f=Forcing(e,s,c);src=B/'B02_渠池归属修复后的参数同化诊断基线/六小时诊断起点';cfg=json.loads((src/'运行配置.json').read_text())
with np.load(src/'末态集合.npz') as z:
 restore(m,{k:z[f'm0_{k}'].copy() for k in STATE_KEYS});theta=z['parameter_log_ratios'][:,0].copy()
for i,p in enumerate(cfg['parameter_bounds']):
 arr=m._n_ik if p['kind']=='n' else m._Co;arr[p['indices']]*=np.exp(theta[i])
m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m.t=21600.;m.iter_count=0
links=c['links'];shape=links['shape'].to_numpy();print('shapes',set(shape),flush=True);ids=np.flatnonzero(np.isin(shape,['rect_open','trapezoidal']));assert len(ids)>0
# Use exactly the runtime linkage and finite geometry height/width; no extension or mutation.
height=links.g1.to_numpy(float)[ids];width=links.g2.to_numpy(float)[ids];slope=np.where(shape[ids]=='trapezoidal',links.g3.to_numpy(float)[ids],0.);length=m._dx_ik[ids];ju=m._Ik[ids];jd=m._Ip1k[ids]
def geometry(h):
 y=np.maximum((h[ju]+h[jd])/2,0);wet=np.minimum(y,height);area=wet*(width+slope*wet);B=width+2*slope*wet
 return y,area,B
link_sum=np.zeros((len(ids),5));rows=[];retries=0;stage=[]
for k in range(36):
 block=np.zeros(5);max_clipped=0
 for t in np.arange(21600+k*1800,21600+(k+1)*1800,15.):
  pending=[(t,15.)]
  while pending:
   tt,dt=pending.pop(0);kwargs=f.at(tt);kwargs.update(dt=dt,num_iter=20,strict=True);before=m.h_Ik.copy();yb,Ab,Bb=geometry(before)
   try:m.step(**kwargs)
   except RuntimeError as err:
    if 'did not converge' not in str(err) or dt<=1.875:raise
    retries+=1;pending[:0]=[(tt,dt/2),(tt+dt/2,dt/2)];continue
   ya,Aa,Ba=geometry(m.h_Ik);dh=(m.h_Ik[ju]-before[ju]+m.h_Ik[jd]-before[jd])/2
   # Width actually used in the implicit continuity coefficients (last solver evaluation).
   discrete=m._B_ik[ids]*length*dh;geometric=(Aa-Ab)*length;gap=discrete-geometric
   clipped=(yb>height)&(ya>height);crossing=(yb>height)^(ya>height);max_clipped=max(max_clipped,int(clipped.sum()))
   contributions=np.column_stack([discrete,geometric,np.where(clipped,gap,0),np.where(~clipped&~crossing,gap,0),np.where(crossing,gap,0)])
   link_sum+=contributions;block+=contributions.sum(axis=0)
 now=21600+(k+1)*1800;rows.append(dict(time_s=now,continuity_width_volume_change_m3=block[0],geometric_volume_change_m3=block[1],stays_above_height_gap_m3=block[2],inside_height_gap_m3=block[3],crossing_height_gap_m3=block[4],max_clipped_links=max_clipped));
 for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st'):
  q=g.iloc[0];stage.append(dict(time_s=now,station=st,forebay_m=m.H_j[int(q.sj_0)],afterbay_m=m.H_j[int(q.sj_1)]))
 if (k+1)%12==0:print('cycles',k+1,flush=True)
pd.DataFrame(rows).to_csv(O/'十八小时逐半小时蓄量差.csv',index=False);pd.DataFrame(stage).to_csv(O/'诊断轨迹水位.csv',index=False)
summary=[]
for j,i in enumerate(ids):summary.append(dict(link_id=int(i),channel=str(c['superlinks'].iloc[int(m._ki[i])]['name']),length_m=float(length[j]),height_m=float(height[j]),discrete_dV_m3=link_sum[j,0],geometry_dV_m3=link_sum[j,1],clipped_gap_m3=link_sum[j,2],inside_gap_m3=link_sum[j,3],crossing_gap_m3=link_sum[j,4]))
l=pd.DataFrame(summary);l.sort_values('clipped_gap_m3',key=abs,ascending=False).to_csv(O/'逐连接蓄量差.csv',index=False)
result=dict(source=str(src),member=0,time_start=21600,time_end=86400,assimilation=False,geometry_changed=False,roughness_or_coefficient_changed_after_start=False,open_internal_links=len(ids),max_concurrent_clipped=max(r['max_clipped_links'] for r in rows),discrete_dV_m3=float(link_sum[:,0].sum()),geometric_dV_m3=float(link_sum[:,1].sum()),clipped_gap_net_m3=float(link_sum[:,2].sum()),clipped_gap_abs_sum_over_output_blocks_m3=float(sum(abs(r['stays_above_height_gap_m3']) for r in rows)),inside_gap_net_m3=float(link_sum[:,3].sum()),crossing_gap_net_m3=float(link_sum[:,4].sum()),retries=retries,wall_seconds=time.time()-start,note='只比较明渠内部连接的连续方程宽度储水项与同连接A*dx变化；不含端段、有压段或节点附加储水，不等同实测漏水/补水或全网水量残差。单成员短段诊断，不代表50成员均值或30天MAE贡献。')
(O/'实际过程蓄量差核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
