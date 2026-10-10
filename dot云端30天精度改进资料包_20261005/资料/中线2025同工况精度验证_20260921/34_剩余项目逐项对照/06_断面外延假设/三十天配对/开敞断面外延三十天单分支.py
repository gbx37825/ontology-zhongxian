"""条件性几何外延量级试验；原始文件/冻结基线不变，只修改内存中的开敞g1。"""
from pathlib import Path
import copy,hashlib,json,sys,time,argparse
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,choices=[0,1],required=True);args=ap.parse_args()
CASE=['原几何固定参数','开敞几何外延假设'][args.case]
O=Path(__file__).resolve().parent/CASE;O.mkdir(exist_ok=False)
R=Path(__file__).resolve().parents[3];F=R/'26_冻结基线与后续归因'
B=F/'B02_渠池归属修复后的参数同化诊断基线'
sys.path.insert(0,str(R/'30_分通道方法逐项测试/03_稳态与分通道试验'))
from 试验水力工具 import build,restore,STATE_KEYS
from 常驻水力进程 import jit_evidence
from pipedream_solver import ngeometry as ng

T0=21600.;END=2592000.;CAP=100.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
engine,source,c,forcing,params=build();base=engine.model
checkpoint=B/'六小时诊断起点/最新完整集合.npz'
clock=json.loads((B/'六小时诊断起点/最新集合时间与随机状态.json').read_text())
assert clock['time_s']==T0
with np.load(checkpoint) as z:
 initial={key:z['m0_'+key].copy() for key in STATE_KEYS}
 theta=z['parameter_log_ratios'][:,0].copy()
assert len(params)==len(theta)==119
old_n=base._n_ik.copy();old_C=base._Co.copy()
for j,p in enumerate(params):
 ids=p['indices'];arr=base._n_ik if p['kind']=='n' else base._Co;reference=old_n if p['kind']=='n' else old_C
 arr[ids]=reference[ids]*np.exp(theta[j])
base._n_uk[:]=base._n_ik[base._i_1k];base._n_dk[:]=base._n_ik[base._i_nk]
restore(base,initial);base.t=T0;base.iter_count=0
open_mask=c['links']['shape'].isin(['rect_open','trapezoidal']).to_numpy();open_ids=np.flatnonzero(open_mask)
original_g1=base._g1_ik.copy();assert original_g1[open_ids].max()<CAP
assert np.isinf(base.max_depth).all() and np.isinf(base.max_depth_k).all()
assert not np.any(base._dx_uk) and not np.any(base._dx_dk)

# Exact geometry values are unchanged at and below the original crown/edge; derivative checked away from the kink.
checks=[]
for row in c['links'].loc[open_ids].itertuples():
 funcs=(ng.Rect_Open_A_ik,ng.Rect_Open_Pe_ik,ng.Rect_Open_B_ik) if row.shape=='rect_open' else (ng.Trapezoidal_A_ik,ng.Trapezoidal_Pe_ik,ng.Trapezoidal_B_ik)
 old_args=(row.g1,row.g2) if row.shape=='rect_open' else (row.g1,row.g2,row.g3)
 new_args=(CAP,)+old_args[1:]
 same=all(fn(y,y,*old_args)==fn(y,y,*new_args) for fn in funcs for y in [0.,row.g1*.25,row.g1*.75,row.g1])
 y=row.g1+.5;eps=1e-5;dA=(funcs[0](y+eps,y+eps,*new_args)-funcs[0](y-eps,y-eps,*new_args))/(2*eps);width=funcs[2](y,y,*new_args)
 assert same and abs(dA-width)<1e-5,(row.id,dA,width)
 checks.append(dict(link_id=int(row.id),within_original_identical=same,derivative_difference=abs(dA-width)))

groups=[]
for station,df in c['orifices'].groupby(c['orifices'].name.str.split('-').str[0]):
 assert df.sj_0.nunique()==df.sj_1.nunique()==1
 groups.extend([(station,'forebay_m',int(df.iloc[0].sj_0)),(station,'afterbay_m',int(df.iloc[0].sj_1))])
assert len(groups)==118
obs=pd.read_csv(F/'共享输入/作者已核实修正观测层.csv')
obs=obs[obs.valid & obs.metric.isin(['forebay_m','afterbay_m']) & (obs.time_s>T0)&(obs.time_s<=END)]
op_points=[]
for name,meta in c['channel_discretization'].items():
 if name not in source['baseData']['channels']:continue
 for p in meta['sections']:op_points.append((p['junction_id'],p['height']))
op_j=np.asarray([p[0] for p in op_points]);op_edge=np.asarray([p[1] for p in op_points])

def volumes(m):
 m.link_hydraulic_geometry();m.compute_storage_volumes()
 return float(m._A_ik[open_ids]@m._dx_ik[open_ids]),float(m._A_ik@m._dx_ik+m.V_j[~m.bc].sum())

def net_external(m,kw):
 bc=m.bc
 net=float(kw['Q_in'][~bc].sum()+kw['Q_0Ik'].sum()+m.Q_uk[bc[m._J_uk]].sum()-m.Q_dk[bc[m._J_dk]].sum())
 net+=float(m.Q_o[bc[m._J_uo]&~bc[m._J_do]].sum()-m.Q_o[~bc[m._J_uo]&bc[m._J_do]].sum())
 assert m.n_w==m.n_p==0
 return net

manifest=dict(scope='数学外延假设总效应，非工程修复、非断面加高建议；不是仅A/B不一致的净贡献',member=0,start_s=T0,end_s=END,no_assimilation=True,parameters_fixed_at_six_hour_checkpoint=True,open_link_count=len(open_ids),closed_links_excluded=int((~open_mask).sum()),fixed_extrapolation_height_m=CAP,node_max_depth_all_infinite=True,within_original_exact_all=all(x['within_original_identical'] for x in checks),max_dA_dh_minus_B_above_original=float(max(x['derivative_difference'] for x in checks)),source_sha256={'checkpoint':sha(checkpoint),'input':sha(F/'共享输入/C07.json'),'observations':sha(F/'共享输入/作者已核实修正观测层.csv'),'script':sha(Path(__file__))})
(O/'试验约定与预检.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
scores=[];audits=[];run_summaries=[]
for case in [CASE]:
 m=copy.deepcopy(base)
 if case=='开敞几何外延假设':m._g1_ik[open_ids]=CAP
 assert all(np.array_equal(getattr(m,k),initial[k]) for k in STATE_KEYS)
 assert np.array_equal(m._g1_ik[~open_mask],base._g1_ik[~open_mask])
 assert np.array_equal(m._g2_ik,base._g2_ik) and np.array_equal(m._g3_ik,base._g3_ik)
 assert np.array_equal(m._n_ik,base._n_ik) and np.array_equal(m._Co,base._Co)
 for key in ['_g1_o','_g2_o','_y_max_o','_z_inv_Ik','_z_inv_j']:assert np.array_equal(getattr(m,key),getattr(base,key))
 m.link_hydraulic_geometry();m.upstream_hydraulic_geometry();m.downstream_hydraulic_geometry();m.compute_storage_volumes();m.save_state()
 vopen0,v0=volumes(m);last_open=vopen0;last_v=v0
 net_volume=0.;disc_volume=0.;open_disc=0.;open_abs_mismatch=0.;retries=0;steps=0;max_head=max_gate=0.;started=time.time();highest_depth=0.
 try:
  for cycle in range(12,1440):
   t0=cycle*1800.;t1=t0+1800.
   for ts in np.arange(t0,t1,15.):
    pending=[(float(ts),15.)]
    while pending:
     tt,dt=pending.pop(0);kw=forcing.at(tt);kw.update(dt=dt,num_iter=20,strict=True)
     hprev=m.h_Ik.copy();Hprev=m.H_j.copy()
     try:m.step(**kw)
     except RuntimeError as ex:
      if 'did not converge' not in str(ex) or dt<=1.875:raise
      retries+=1;pending[:0]=[(tt,dt/2),(tt+dt/2,dt/2)];continue
     assert all(np.isfinite(getattr(m,k)).all() for k in STATE_KEYS)
     dh=m.h_Ik-hprev
     Bstep=m._B_ik.copy()
     d_open=float(np.sum(.5*Bstep[open_ids]*m._dx_ik[open_ids]*(dh[m._Ik[open_ids]]+dh[m._Ip1k[open_ids]])))
     d_disc=float(m._E_Ik@dh*dt+m._A_sj[~m.bc]@(m.H_j[~m.bc]-Hprev[~m.bc]))
     net_volume+=net_external(m,kw)*dt;disc_volume+=d_disc;open_disc+=d_open
     vopen,v=volumes(m);open_abs_mismatch+=abs((vopen-last_open)-d_open);last_open=vopen;last_v=v
     steps+=1;max_head=max(max_head,m.last_head_residual);max_gate=max(max_gate,m.last_gate_flow_residual)
     mean_depth=.5*(m.h_Ik[m._Ik]+m.h_Ik[m._Ip1k]);highest_depth=max(highest_depth,float(mean_depth[open_ids].max()))
     assert highest_depth<CAP-1.,'假设上限触发，停止而非进一步加高'
   for station,metric,j in groups:scores.append(dict(case=case,time_s=t1,station_code=station,metric=metric,sim=float(m.H_j[j])))
   mean_depth=.5*(m.h_Ik[m._Ik]+m.h_Ik[m._Ip1k])
   audit=dict(case=case,time_s=t1,cycle=cycle+1,original_contour_exceed_nodes=int((m.h_Ik[op_j]>op_edge+1e-6).sum()),original_cap_exceed_links=int((mean_depth[open_ids]>original_g1[open_ids]+1e-6).sum()),maximum_above_original_m=float(np.maximum(mean_depth[open_ids]-original_g1[open_ids],0).max()),open_volume_m3=vopen,full_nonfixed_volume_m3=v,open_geometric_change_m3=vopen-vopen0,open_B_storage_change_m3=open_disc,open_geometric_minus_B_m3=vopen-vopen0-open_disc,cumulative_abs_open_geometry_B_mismatch_m3=open_abs_mismatch,net_external_volume_m3=net_volume,discrete_volume_change_m3=disc_volume,net_minus_discrete_m3=net_volume-disc_volume,net_minus_geometry_m3=net_volume-(v-v0),retries=retries)
   audits.append(audit)
   if (cycle+1)%24==0:
    print(json.dumps({**audit,'wall_seconds':time.time()-started},ensure_ascii=False),flush=True)
    (O/'进度.json').write_text(json.dumps(dict(case=case,completed_cycle=cycle+1,total_cycle=1440,status='RUNNING'),ensure_ascii=False))
  assert m.t==END
  assert np.array_equal(m._n_ik,base._n_ik) and np.array_equal(m._Co,base._Co)
  np.savez_compressed(O/(case+'_末态.npz'),**{k:getattr(m,k) for k in STATE_KEYS},parameter_log_ratios=theta)
  run_summaries.append(dict(case=case,status='COMPLETED',cycles=1428,steps=steps,retries=retries,initial_open_volume_m3=vopen0,initial_full_nonfixed_volume_m3=v0,last_audit=audits[-1],highest_open_mean_depth_m=highest_depth,max_head_residual_m=max_head,max_gate_residual_m3s=max_gate,wall_s=time.time()-started,numba=jit_evidence(),parameters_unchanged=True,solver_identity=engine.solver_identity))
 except BaseException as ex:
  (O/'失败.json').write_text(json.dumps(dict(case=case,time_s=float(m.t),error=repr(ex),retries=retries),ensure_ascii=False,indent=2));raise

df=pd.DataFrame(scores).merge(obs[['station_code','metric','time_s','value','pdf_comparable']],on=['station_code','metric','time_s'],how='left',validate='many_to_one')
assert df.value.notna().all() and len(df)==1428*118
df['error']=df.sim-df.value
windows=[('第6小时至第30天',T0,END),('第3天后至第30天',259200.,END),('第6小时至第3天',T0,259200.)]
metrics=[];points=[]
for label,lo,hi in windows:
 for case in df.case.unique():
  part=df[(df.case==case)&(df.time_s>lo)&(df.time_s<=hi)]
  for scope in ['118个水位测点','作者88个可比水位测点']:
   p=part if scope.startswith('118') else part[part.pdf_comparable]
   worst=p.loc[p.error.abs().idxmax()]
   metrics.append(dict(window=label,case=case,scope=scope,measurements=len(p),points=p[['station_code','metric']].drop_duplicates().shape[0],mae_m=float(p.error.abs().mean()),rmse_m=float(np.sqrt((p.error**2).mean())),max_abs_m=float(p.error.abs().max()),worst_station=worst.station_code,worst_metric=worst.metric,worst_time_s=float(worst.time_s)))
  if label=='第6小时至第30天':
   for (st,met),p in part.groupby(['station_code','metric']):points.append(dict(case=case,station_code=st,metric=met,mae_m=float(p.error.abs().mean()),max_abs_m=float(p.error.abs().max())))
summary=dict(**manifest,status='COMPLETED',runs=run_summaries,metrics=metrics)
(O/'三十天单分支结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
df.to_csv(O/'水位与实测.csv.gz',index=False,compression='gzip');pd.DataFrame(audits).to_csv(O/'几何与储水逐半小时.csv',index=False);pd.DataFrame(points).to_csv(O/'逐测点精度.csv',index=False);pd.DataFrame(metrics).to_csv(O/'误差指标.csv',index=False)
(O/'进度.json').write_text(json.dumps(dict(status='COMPLETED',completed_cycle=1440,simulated_cycles_per_case=1428),ensure_ascii=False))
print(json.dumps(dict(status='COMPLETED',metrics=metrics),ensure_ascii=False),flush=True)
