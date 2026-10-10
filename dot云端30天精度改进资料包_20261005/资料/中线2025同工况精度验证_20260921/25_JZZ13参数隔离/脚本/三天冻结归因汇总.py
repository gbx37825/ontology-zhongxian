from pathlib import Path
import json,numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];R=P.parent;B=R/'26_冻结基线与后续归因/B02_渠池归属修复后的参数同化诊断基线';source=B/'六小时诊断起点';checks=[];summ=[];rejections=[];params=[]
runs=[('正常同化',B/'同化运行',None),('冻结沙河闸流系数',P/'G05_冻结沙河系数至第3天','G03_只冻结沙河闸流系数'),('冻结第12渠池糙率',P/'G06_冻结第12渠池糙率至第3天','G04_只冻结第12渠池糙率')]
periods=[('第6—24小时',21600,86400),('第2天',86400,172800),('第3天',172800,259200)]
for label,d,first in runs:
 cfg=json.loads((d/'运行配置.json').read_text());all_scores=pd.read_csv(d/'预报与分析逐点.csv')
 if first:
  st=json.loads((d/'进度.json').read_text());assert st['status']=='COMPLETED' and st['completed_cycles']==144 and not (d/'失败.json').exists()
  assert not all_scores.duplicated(['time_s','phase','operator','station','metric']).any()
  for k,g in all_scores.groupby(['phase','operator','station','metric']):assert np.array_equal(np.sort(g.time_s),np.arange(1,145)*1800)
  par=json.loads((d/'隔离分支.json').read_text())['frozen'];j=next(i for i,p in enumerate(cfg['parameter_bounds']) if p['name']==par)
  with np.load(source/'末态集合.npz') as a,np.load(d/'末态集合.npz') as z:
   assert len(z.files)==301 and z['parameter_log_ratios'].shape==(119,50) and all(np.isfinite(z[k]).all() for k in z.files)
   assert all(f'm{i}_{k}' in z for i in range(50) for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o'])
   assert np.array_equal(a['parameter_log_ratios'][j],z['parameter_log_ratios'][j])
   ratio=np.exp(a['parameter_log_ratios'][j]);reference=[ratio.mean(),ratio.min(),ratio.max(),a['parameter_log_ratios'][j].std(ddof=1)]
  track=pd.read_csv(d/'参数轨迹.csv');assert len(track)==144*119 and not track.duplicated(['time_s','parameter']).any();fr=track[(track.parameter==par)&(track.time_s>=21600)]
  for col,v in zip(['mean_ratio','min_ratio','max_ratio','std_log'],reference):assert np.allclose(fr[col],v,rtol=0,atol=1e-14)
  ledger=pd.read_csv(d/'同化等效水量.csv');assert np.array_equal(ledger.time_s,np.arange(1,145)*1800) and ledger.state_unchanged.all() and (ledger.analysis_delta_volume_mean_m3==0).all()
  jit=json.loads((d/'Numba实际执行证据.json').read_text());assert len(jit['workers'])==4 and all(not w['jit_disabled'] and w['momentum_compiled_signatures']>0 and w['gate_compiled_signatures']>0 for w in jit['workers'])
  for file in ['预报与分析逐点.csv','参数轨迹.csv','同化等效水量.csv','逐观测更新记录.csv']:
   a=pd.read_csv(P/first/file);b=pd.read_csv(d/file);b=b[b.time_s<=86400].reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
  phases={k:g.drop(columns='phase').reset_index(drop=True) for k,g in all_scores.groupby('phase')};pd.testing.assert_frame_equal(phases['forecast_before_update'],phases['after_parameter_update_same_state'],check_exact=True)
  checks.append(dict(run=d.name,cycles=144,members=50,parameters=119,freeze_parameter=par,final_frozen_members_exact=True,trajectory_summaries_unchanged=True,note='逐周期保存的是均值极值及离散度，完整逐成员仅起末态；代码冻结分支不写该参数，未声称有每周期完整成员存档',first_day_exact=True,instantaneous_volume_adjustment=0,state_unchanged=True,numba_workers=4,status=st))
 f=all_scores[all_scores.phase=='forecast_before_update'].copy();f['error']=f.sim-f.observed;log=pd.read_csv(d/'逐观测更新记录.csv');track=pd.read_csv(d/'参数轨迹.csv')
 for period,lo,hi in periods:
  t=f[(f.time_s>lo)&(f.time_s<=hi)];scopes=[('全线118个水位测点',t[t.operator=='闸端全线']),('9个检验水位测点',t[t.group=='空间留出点'])]
  for code,name in [('JZZ12','澎河'),('JZZ13','沙河'),('JZZ14','玉带河')]:
   for met,title in [('forebay_m','闸前'),('afterbay_m','闸后')]:scopes.append((name+title+'模型闸端',t[(t.operator=='闸端全线')&(t.station==code)&(t.metric==met)]))
   scopes.append((name+'闸后实际测点',t[(t.operator=='同测点位置')&(t.station==code)&(t.metric=='afterbay_m')]))
  for scope,g in scopes:
   if len(g):summ.append(dict(run=label,period=period,scope=scope,n=len(g),mae=abs(g.error).mean(),max_error=abs(g.error).max(),bias=g.error.mean()))
  l=log[(log.time_s>lo)&(log.time_s<=hi)]
  for scope,z in [('全部更新观测',l)]+[(name,l[l.station==code]) for code,name in [('JZZ12','澎河'),('JZZ13','沙河'),('JZZ14','玉带河')]]:rejections.append(dict(run=label,period=period,scope=scope,count=len(z),accepted=int((z.status=='used_for_parameters').sum()),rejected=int((z.status=='rejected_innovation').sum())))
  for p in ['POOL12_n','POOL13_n','JZZ12_C','JZZ13_C','JZZ14_C']:
   g=track[(track.parameter==p)&(track.time_s==hi)].iloc[0];params.append(dict(run=label,period=period,parameter=p,mean_ratio=g.mean_ratio,min_ratio=g.min_ratio,max_ratio=g.max_ratio,std_log=g.std_log))
s=pd.DataFrame(summ);s.to_csv(P/'三天冻结分时段精度.csv',index=False);pd.DataFrame(rejections).to_csv(P/'三天冻结拒收对照.csv',index=False);pd.DataFrame(params).to_csv(P/'三天冻结参数轨迹对照.csv',index=False);(P/'三天冻结完成核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
lines=['# 沙河单参数冻结三天归因报告','','## 核验与比较范围','','两组均完成144周期至第3天，50成员六类水力状态及119参数有限，无失败、无减步重试。共同前6小时来自冻结B02检查点，真正分支区间为第6—72小时。原B01、B02均未修改。首日逐点水位、参数轨迹、水量及观测更新记录分别与G03/G04逐元素一致。指定冻结参数的50成员末值等于起点，全程保存的均值、极值及离散度不变；参数更新不改水力状态，即时蓄水调整全部0。4个工作进程Numba编译执行已核实。','','比较基准是冻结B02的正常参数同化，同时间段、同观测口径，不另跑无同化。下面的冻结效应包含后续状态及其他参数响应，是闭环总效应，不是瞬时偏导。','','## 分时段误差','','每格为MAE / 最大绝对误差，单位m。模型闸端与实际安装位置算子分别列示。','','|时段|测点范围|正常同化|冻结沙河闸流系数|冻结第12渠池糙率|','|---|---|---:|---:|---:|']
for period,lo,hi in periods:
 for scope in s.scope.unique():
  vals=[]
  for label,d,_ in runs:
   g=s[(s.run==label)&(s.period==period)&(s.scope==scope)]
   vals.append('无评分记录' if g.empty else f'{g.iloc[0].mae:.6f} / {g.iloc[0].max_error:.6f}')
  lines.append('|'+period+'|'+scope+'|'+'|'.join(vals)+'|')
lines+=['','## 观测拒收','','|时段|范围|正常同化拒收/候选|冻结沙河系数拒收/候选|冻结第12渠池糙率拒收/候选|','|---|---|---:|---:|---:|']
rej=pd.DataFrame(rejections)
for period,_,_ in periods:
 for scope in rej.scope.unique():
  vals=[]
  for label,_,_ in runs:
   q=rej[(rej.run==label)&(rej.period==period)&(rej.scope==scope)].iloc[0];vals.append(f'{q.rejected}/{q["count"]}')
  lines.append('|'+period+'|'+scope+'|'+'|'.join(vals)+'|')
lines+=['','## 参数轨迹','','均值倍率相对于原模型参数，冻结值是第6小时各成员值，不是统一重置到1。完整极值和离散度见三天冻结参数轨迹对照.csv。','','|时段|参数|正常同化|冻结沙河系数|冻结第12渠池糙率|','|---|---|---:|---:|---:|']
t=pd.DataFrame(params)
for period,_,_ in periods:
 for par in t.parameter.unique():lines.append('|'+period+'|'+par+'|'+'|'.join(f'{t[(t.run==label)&(t.period==period)&(t.parameter==par)].iloc[0].mean_ratio:.6f}' for label,_,_ in runs)+'|')
lines+=['','## 水量、测点与结论边界','','此前第6—24小时澎河至玉带河的条件性水量核查，扣除分水与估算蓄量后剩余约0.233m³/s；蓄量来自断面及水位插值，未独立量测，不能据此认定流量计正确或给单闸补流。沙河安装资料与模型连接闸位相差约788m，仍未闭合；按相对闸距试取水位在第6小时和第30天末只改变厘米量级，不能直接解释米级误差。详见单参数冻结与水量测点核查.md。','','本试验仅3天，未覆盖已知第6天、第11天起的长期全拒收阶段，不能证明该问题解决。没有改观测位置、加流量、取消拒收或放宽收敛门槛。全部使用历史已知边界，不能据短时收益宣布30天或独立未来预报有效。']
(P/'三天单参数冻结归因报告.md').write_text('\n'.join(lines)+'\n');print(s[s.scope.isin(['沙河闸前模型闸端','沙河闸后实际测点','9个检验水位测点'])].to_string(index=False));print(rej[rej.scope=='全部更新观测'].to_string(index=False))
