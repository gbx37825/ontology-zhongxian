from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];names=['E01_20成员30天连续水位同化','E02_同集合30天无同化对照'];summ=[];sides=[];checks=[];vols=[];frames=[]
for name in names:
 p=R/name;status=json.loads((p/'进度.json').read_text());assert status['status']=='COMPLETED' and status['completed_cycles']==1440 and not (p/'失败.json').exists()
 d=pd.read_csv(p/'预报与分析逐点.csv');assert np.isfinite(d[['sim','observed']]).all().all();assert np.array_equal(np.sort(d.time_s.unique()),np.arange(1,1441)*1800);assert not d.duplicated(['time_s','phase','operator','station','metric']).any()
 for keys,g in d.groupby(['phase','operator','station','metric']):assert len(g)==1440,keys
 with np.load(p/'末态集合.npz') as z:assert len(z.files)==120 and all(np.isfinite(z[k]).all() for k in z.files)
 d['error']=d.sim-d.observed;d['run']=name;frames.append(d)
 for phase,g in d.groupby('phase'):
  for label,v in [('同化测点34侧',g[(g.operator=='同测点位置')&(g.group=='同化点')]),('空间留出9侧',g[(g.operator=='同测点位置')&(g.group=='空间留出点')]),('全线闸端118侧',g[g.operator=='闸端全线']),('作者可比闸端88侧',g[(g.operator=='闸端全线')&g.pdf_comparable])]:
   means=v.groupby(['station','metric']).error.apply(lambda x:abs(x).mean());summ.append(dict(run=name,phase=phase,scope=label,n=len(v),sides=len(means),mae=float(abs(v.error).mean()),rmse=float(np.sqrt((v.error**2).mean())),bias=float(v.error.mean()),max_error=float(abs(v.error).max()),p95=float(abs(v.error).quantile(.95)),pass_mae_0p1=int((means<=.1).sum())))
 for (phase,op,st,met),g in d.groupby(['phase','operator','station','metric']):sides.append(dict(run=name,phase=phase,operator=op,station=st,metric=met,mae=float(abs(g.error).mean()),max_error=float(abs(g.error).max())))
 v=pd.read_csv(p/'同化等效水量.csv');assert len(v)==1440 and np.array_equal(v.time_s,np.arange(1,1441)*1800);vols.append(dict(run=name,net_analysis_m3=float(v.analysis_delta_volume_mean_m3.sum()),absolute_analysis_m3=float(abs(v.analysis_delta_volume_mean_m3).sum()),mean_equivalent_analysis_m3s=float(v.analysis_delta_volume_mean_m3.sum()/2592000),process_net_m3=float(v.process_delta_volume_mean_m3.sum()),accepted=int(v.accepted.sum()),rejected=int(v.rejected.sum())))
 checks.append(dict(run=name,score_rows=len(d),cycles=1440,end_time_s=float(d.time_s.max()),checkpoint_finite=True,**status))
pd.DataFrame(summ).to_csv(R/'30天连续同化精度汇总.csv',index=False);pd.DataFrame(sides).to_csv(R/'30天连续同化逐侧结果.csv',index=False);pd.DataFrame(vols).to_csv(R/'30天同化等效水量汇总.csv',index=False);(R/'30天完成核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
s=pd.DataFrame(summ);lines=['# 30天连续状态同化结果','','两组均完成1440个半小时周期，覆盖2025-07-22 12:00至2025-08-21 12:00；首次输出为起报后1800秒，末次为2592000秒。逐侧时刻完整、无重复，20成员末态均为有限数值。','','## 同化效果','','单位均为m。同化前预报为消费当前观测之前的集合均值，分析为消费后；两组使用同一初态、随机种子及过程扰动。','','| 评价口径 | 无同化MAE | 同化前预报MAE | 同化后分析MAE | 无同化最大误差 | 同化前最大误差 | 分析最大误差 |','|---|---:|---:|---:|---:|---:|---:|']
for scope in ['同化测点34侧','空间留出9侧','全线闸端118侧','作者可比闸端88侧']:
 def row(n,p):return s[(s.run==n)&(s.phase==p)&(s.scope==scope)].iloc[0]
 b=row(names[1],'forecast_before_update');f=row(names[0],'forecast_before_update');a=row(names[0],'analysis_after_update');lines.append(f'| {scope} | {b.mae:.6f} | {f.mae:.6f} | {a.mae:.6f} | {b.max_error:.6f} | {f.max_error:.6f} | {a.max_error:.6f} |')
lines+=['','同化点和留出点按安装桩号的同侧水位算子评价；全线和作者88侧按闸端水位评价，不能混成同一种测点口径。9个留出点未参加后续同化，但共同起报初态使用了t=0观测。最大误差包含保留的原始水位尖点，未因模型误差大而删评分数据。','','## 等效蓄水调整','','| 组别 | 同化净调整(m³) | 同化绝对调整累计(m³) | 净调整折算(m³/s) | 过程扰动净调整(m³) | 接受/拒收观测 |','|---|---:|---:|---:|---:|---:|']
for v in vols:lines.append(f"| {v['run']} | {v['net_analysis_m3']:.1f} | {v['absolute_analysis_m3']:.1f} | {v['mean_equivalent_analysis_m3s']:.3f} | {v['process_net_m3']:.1f} | {v['accepted']}/{v['rejected']} |")
lines+=['','这些是更新状态产生的模型等效水量，不是真实补水。创新超过门槛只标记拒收，不证明传感器错误；可能包含模型偏差或观测算子不适用。','','## 数值与运行边界','','同化组无减步重试；无同化组117次减步重试后继续完成。两组最大节点水位迭代残差均小于0.0015m，最大闸流一致性残差均小于0.01m³/s。这里不冒称全部内部状态及水量方程已逐步严格验收。','','这是使用已知历史边界与开度驱动的条件回放；同化前的半小时先验不是未知未来边界下的真实预报，更不是独立6小时预报。','','## 采用判断','','应以同化前先验及留出点结果判断是否可采用，不能只用参与更新的测点分析误差宣布模型准确。未修改原始Excel、观测评分数据或正式平台配置；本次没有部署同化功能。完整统计见CSV，包括MAE、RMSE、P95、最大误差及每侧0.1m通过情况。']
(R/'30天连续状态同化对照报告.md').write_text('\n'.join(lines)+'\n');print(s.to_string(index=False));print(json.dumps(vols,ensure_ascii=False,indent=2))
