from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];R=P.parent
names=['P01M4_50成员30天参数同化','P02M4_50成员30天无同化'];summ=[];checks=[];trajectories=[];frames=[];side_rows=[]
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
old=pd.read_csv(R/'20_连续状态同化/30天连续同化精度汇总.csv');old=old[old.phase=='forecast_before_update']
lines=['# 50成员30天参数同化对照报告','','## 完成与评价口径','','两组均完成1440个半小时周期，模拟覆盖2025年7月22日12:00至8月21日12:00；评分输出从起报后半小时至第30天，逐侧1440条，无缺时或重复。50成员六类水力状态及119×50参数末态均为有限数值，最新检查点与末态完全一致。观测评分值已回查作者已核实修正观测层.csv。','','本次只更新明渠糙率和闸门流量系数，同化前的水位用于评分。同一时刻参数更新后水位与更新前完全相同，没有新的水位拟合分析结果。34侧参与在线参数更新，9侧空间留出不参与在线更新；但起报初态仍使用既有观测约束。测点位置口径与闸端口径分开统计。','','## 精度对照','','单位m，最大误差指所有有效时刻中的最大绝对误差。','','|范围|无同化MAE|参数同化MAE|MAE改善|无同化RMSE|参数同化RMSE|无同化最大误差|参数同化最大误差|同化后逐侧月MAE≤0.1m|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for scope in s.scope.unique():
 base=s[(s.run==names[1])&(s.scope==scope)].iloc[0];v=s[(s.run==names[0])&(s.scope==scope)].iloc[0]
 lines.append(f'|{scope}|{base.mae:.6f}|{v.mae:.6f}|{(1-v.mae/base.mae)*100:.1f}%|{base.rmse:.6f}|{v.rmse:.6f}|{base.max_error:.6f}|{v.max_error:.6f}|{v.pass_mae_0p1}/{v.sides}|')
lines+=['','## 与此前20成员状态同化比较','','仅比较消费当前观测之前的先验。成员数、扰动方式及更新对象均不同，不是纯单因素算法排名。此前的同化后分析精度不作为本次参数同化的先验对手。','','|范围|20成员无同化MAE|20成员状态同化先验MAE|50成员无同化MAE|50成员参数同化先验MAE|','|---|---:|---:|---:|---:|']
for scope in s.scope.unique():
 vals=[old[(old.scope==scope)&(old.run.str.startswith(code))].iloc[0].mae for code in ['E02','E01']]+[s[(s.scope==scope)&(s.run==n)].iloc[0].mae for n in names[::-1]]
 lines.append('|'+scope+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
lines+=['','## 水量与参数核验','','两组每次参数更新前后state_unchanged均为真，即时蓄水调整均为0；没有水力状态过程噪声。参数改变仍会在后续积分中影响输水与蓄水过程，因此即时调整为0不等于全程水量平衡已独立验收。此前20成员状态同化净等效蓄水调整约1096.94万m³，本次在线参数更新不直接增减这部分水量。','','|组别|接受观测|大创新拒收|触及下界的成员×参数×周期数|触及上界的成员×参数×周期数|末步处于边界的成员×参数数|参数离散度补充次数（参数×周期）|','|---|---:|---:|---:|---:|---:|---:|']
for c in checks:
 t=tr[tr.run==c['run']];lines.append(f"|{c['run']}|{c['accepted']}|{c['rejected']}|{t.lower_member_cycles.sum()}|{t.upper_member_cycles.sum()}|{(t.final_lower+t.final_upper).sum()}|{c['rejuvenated_parameter_cycles']}|")
lines+=['','轨迹及末态均未越过配置的物理参数边界。触界次数不是不同参数的个数；逐参数起末倍率、极值、触界及离散度见30天参数轨迹与触界汇总.csv。1%限制针对同化分析增量；离散度补充是另一步，不能把完整参数轨迹相邻差都解释为受1%约束。大创新拒收只代表本次没有采用该观测，不证明传感器坏了，也可能是模型偏差或测点代表性问题。','','## 最大误差位置','','|组别|范围|测点|时距(s)|最大误差(m)|','|---|---|---|---:|---:|']
for v in summ:lines.append(f"|{v['run']}|{v['scope']}|{v['worst_station']} {v['worst_metric']}|{v['worst_time_s']:.0f}|{v['max_error']:.6f}|")
lines+=['','所有既定有效观测均保留评分，未为改善指标删除大误差。','','## 并行、断点与运行边界','','每组4个常驻工作进程，半小时同步后主进程统一同化；各工作进程有Numba实际编译执行证据。6小时串并行完整末态、参数、随机数及观测记录严格一致，水力推进短测加速2.64倍；切换重叠段也通过核验。恢复保留50成员、119参数与随机数，并拼接断点前记录，不重复计分。','','最终运行日志的耗时和重试计数仅对应第8天至第30天的恢复段，不是从零开始全程成本。']
for c in checks:
 st=c['resumed_segment_status'];lines.append(f"- {c['run']}：恢复段{st['wall_seconds']:.1f}秒，减步重试{st['retries']}次；最大水位迭代残差{st['max_head_residual']:.9f}m，闸流一致性残差{st['max_gate_residual']:.9f}m³/s。")
lines+=['','这是已知历史边界及闸门开度驱动的回放，不是未知未来输入的独立预报；此处半小时先验精度也不能替代独立6小时预报检验。局地化仍使用本试验20km方式，未改为论文分区。原始资料、旧试验和平台部署均未覆盖。']
(P/'50成员30天参数同化对照报告.md').write_text('\n'.join(lines)+'\n');print(s.to_string(index=False));print(json.dumps(checks,ensure_ascii=False))
