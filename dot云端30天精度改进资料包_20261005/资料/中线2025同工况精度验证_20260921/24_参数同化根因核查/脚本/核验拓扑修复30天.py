from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];R=P.parent
names=['F03_拓扑分组修复30天','F04_拓扑分组匹配无同化30天'];summ=[];checks=[];trajectories=[];frames=[];side_rows=[]
obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');obs=obs[obs.valid].set_index(['station_code','metric','time_s'])
times=np.arange(1,1441)*1800
for name in names:
 d=P/name;status=json.loads((d/'进度.json').read_text());config=json.loads((d/'运行配置.json').read_text());assert status['status']=='COMPLETED' and status['completed_cycles']==1440 and not (d/'失败.json').exists();assert config['members']==50 and config['parameter_count']==119
 scores=pd.read_csv(d/'预报与分析逐点.csv');assert np.isfinite(scores[['sim','observed']]).all().all();assert not scores.duplicated(['time_s','phase','operator','station','metric']).any()
 for keys,g in scores.groupby(['phase','operator','station','metric']):assert np.array_equal(np.sort(g.time_s),times),keys
 idx=pd.MultiIndex.from_arrays([scores.station,scores.metric,scores.time_s]);np.testing.assert_allclose(scores.observed,obs.loc[idx,'value'],rtol=0,atol=1e-12)
 phases={k:g.drop(columns='phase').reset_index(drop=True) for k,g in scores.groupby('phase')};pd.testing.assert_frame_equal(phases['forecast_before_update'],phases['after_parameter_update_same_state'],check_exact=True)
 with np.load(d/'末态集合.npz') as z:
  assert len(z.files)==301 and z['parameter_log_ratios'].shape==(119,50) and all(np.isfinite(z[k]).all() for k in z.files)
  for i in range(50):assert all(f'm{i}_{k}' in z for k in ('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o'))
  theta=z['parameter_log_ratios'].copy()
 stamp=json.loads((d/'最新集合时间与随机状态.json').read_text());assert stamp['cycle']==1440 and stamp['time_s']==2592000
 with np.load(d/'最新完整集合.npz') as a,np.load(d/'末态集合.npz') as b:
  assert all(np.array_equal(a[k],b[k]) for k in b.files)
 v=pd.read_csv(d/'同化等效水量.csv');assert np.array_equal(v.time_s,times) and v.state_unchanged.all() and (v.analysis_delta_volume_mean_m3==0).all()
 jit=json.loads((d/'Numba实际执行证据.json').read_text());cfg=json.loads((d/'计算进程配置.json').read_text());assert len(jit['workers'])==cfg['workers']==4
 assert all(not x['jit_disabled'] and x['momentum_compiled_signatures']>0 and x['gate_compiled_signatures']>0 for x in jit['workers'])
 assert [x['pid'] for x in cfg['worker_evidence']]==[x['pid'] for x in jit['workers']]
 track=pd.read_csv(d/'参数轨迹.csv');assert len(track)==1440*119 and not track.duplicated(['time_s','parameter']).any();assert np.isfinite(track.select_dtypes('number')).all().all()
 for j,par in enumerate(config['parameter_bounds']):
  g=track[track.parameter==par['name']];assert np.array_equal(g.time_s,times)
  assert (g.min_ratio>=par['lower']-1e-10).all() and (g.max_ratio<=par['upper']+1e-10).all()
  assert (theta[j]>=np.log(par['lower'])-1e-10).all() and (theta[j]<=np.log(par['upper'])+1e-10).all()
  trajectories.append(dict(run=name,parameter=par['name'],kind=par['kind'],first_mean=g.mean_ratio.iloc[0],final_mean=g.mean_ratio.iloc[-1],minimum=g.min_ratio.min(),maximum=g.max_ratio.max(),lower_bound=par['lower'],upper_bound=par['upper'],lower_member_cycles=int(g.at_lower.sum()),upper_member_cycles=int(g.at_upper.sum()),final_lower=int(g.at_lower.iloc[-1]),final_upper=int(g.at_upper.iloc[-1]),final_std_log=g.std_log.iloc[-1]))
 accepted=int(v.accepted.sum());rejected=int(v.rejected.sum())
 if not config['no_analysis']:
  log=pd.read_csv(d/'逐观测更新记录.csv');assert len(log)==34*1440 and not log.duplicated(['time_s','station','metric']).any();assert (log.status=='used_for_parameters').sum()==accepted and (log.status=='rejected_innovation').sum()==rejected
  rej=log[log.status=='rejected_innovation'];rej.groupby(['station','metric']).agg(rejected=('status','size'),max_abs_innovation=('innovation',lambda s:abs(s).max())).sort_values('rejected',ascending=False).to_csv(P/'大创新拒收逐测点统计.csv')
 else:assert accepted==rejected==0
 f=scores[scores.phase=='forecast_before_update'].copy();f['error']=f.sim-f.observed;frames.append(f)
 for label,g,expected in [('同化测点34侧',f[(f.operator=='同测点位置')&(f.group=='同化点')],34),('空间留出9侧',f[(f.operator=='同测点位置')&(f.group=='空间留出点')],9),('全线闸端118侧',f[f.operator=='闸端全线'],118),('作者可比闸端88侧',f[(f.operator=='闸端全线')&f.pdf_comparable],88)]:
  means=g.groupby(['station','metric']).error.apply(lambda s:abs(s).mean());assert len(means)==expected and len(g)==expected*1440
  worst=g.loc[g.error.abs().idxmax()];summ.append(dict(run=name,scope=label,n=len(g),sides=len(means),mae=abs(g.error).mean(),rmse=np.sqrt((g.error**2).mean()),bias=g.error.mean(),max_error=abs(g.error).max(),pass_mae_0p1=int((means<=.1).sum()),worst_station=worst.station,worst_metric=worst.metric,worst_time_s=worst.time_s))
 for key,g in f.groupby(['operator','station','metric']):side_rows.append(dict(run=name,operator=key[0],station=key[1],metric=key[2],mae=abs(g.error).mean(),rmse=np.sqrt((g.error**2).mean()),max_error=abs(g.error).max()))
 checks.append(dict(run=name,cycles=1440,score_rows=len(scores),start_output_s=1800,end_output_s=2592000,complete_members=50,parameters=119,state_unchanged_all=True,instantaneous_volume_adjustment_m3=0,pre_post_water_level_exact=True,observations_verified=True,numba_worker_count=4,accepted=accepted,rejected=rejected,rejuvenated_parameter_cycles=int(v.rejuvenated_parameters.sum()),resumed_segment_status=status,final_sha256=hashlib.sha256((d/'末态集合.npz').read_bytes()).hexdigest()))
keys=['time_s','operator','station','metric'];a,b=[x.set_index(keys).sort_index() for x in frames];assert a.index.equals(b.index);pd.testing.assert_series_equal(a.observed,b.observed)
s=pd.DataFrame(summ);s.to_csv(P/'30天参数同化精度汇总.csv',index=False);pd.DataFrame(side_rows).to_csv(P/'30天参数同化逐侧精度.csv',index=False);tr=pd.DataFrame(trajectories);tr.to_csv(P/'30天参数轨迹与触界汇总.csv',index=False)
(P/'30天参数同化完成核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))

print(s.to_string(index=False));print(json.dumps(checks,ensure_ascii=False))
