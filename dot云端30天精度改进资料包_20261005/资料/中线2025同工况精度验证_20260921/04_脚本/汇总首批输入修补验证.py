from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'14_输入修补交付';baseline='B00_实际开度同工况30天';candidate='Q03_六点开度修补同工况30天';obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');stats=[];flows=[];series=[];initial={}
for run in [baseline,candidate]:
 p=R/'06_结果'/run;status=json.loads((p/'运行摘要.json').read_text());assert status['status']=='COMPLETED' and status['completed_steps']==172800,status
 raw=pd.read_csv(p/'逐闸水位流量.csv');initial[run]=raw[raw.time_s==0].set_index('station_code');d=raw.melt(id_vars=['station_code','time_s'],var_name='metric',value_name='sim').merge(obs,on=['station_code','time_s','metric'],validate='one_to_one');d=d[d.valid&(d.metric!='flow_m3s')&(d.time_s>0)];d['raw_error']=d.sim-d.raw_value;d['author_error']=d.sim-d.value
 for period,lo,hi in [('全30天',0,2592000),('首6小时',0,21600),('济河处理窗后',1999800,2592000)]:
  v=d[(d.time_s>lo)&(d.time_s<=hi)]
  for area,g in [('全线118侧',v),('作者88侧',v[v.pdf_comparable]),('七里河两侧',v[v.station_code=='JZZ42']),('济河两侧',v[v.station_code=='JZZ27'])]:
   for basis,col in [('原始观测','raw_error'),('作者已核实修正','author_error')]:
    errs=g[col];maes=g.groupby(['station_code','metric'])[col].apply(lambda x:abs(x).mean());stats.append(dict(run=run,period=period,area=area,basis=basis,n=len(g),mae=float(abs(errs).mean()),max_error=float(abs(errs).max()),pass_mae_0p1=int((maes<=.1).sum()),sides=len(maes)))
 for (station,metric),g in d.groupby(['station_code','metric']):series.append(dict(run=run,station=station,metric=metric,mae=float(abs(g.author_error).mean()),raw_mae=float(abs(g.raw_error).mean())))
 f=raw[raw.time_s>0];flows.append(dict(run=run,output_interval_s=1800,negative_station_frames=int((f.flow_m3s<0).sum()),min_station_flow=float(f.flow_m3s.min()),max_station_flow=float(f.flow_m3s.max()),negative_stations=sorted(f.loc[f.flow_m3s<0,'station_code'].unique())))
a=initial[baseline];b=initial[candidate];initial_diff=float(abs(a.sort_index()[['forebay_m','afterbay_m','flow_m3s']]-b.sort_index()[['forebay_m','afterbay_m','flow_m3s']]).to_numpy().max());assert initial_diff<1e-10,initial_diff
pd.DataFrame(stats).to_csv(O/'首批修补精度对照.csv',index=False);pd.DataFrame(series).to_csv(O/'首批修补逐侧对照.csv',index=False)
manifest=dict(candidate_run=candidate,candidate_input_sha256=json.loads((R/'06_结果'/candidate/'运行摘要.json').read_text())['model_sha256'],baseline_run=baseline,initial_max_difference=initial_diff,completed_days=30,flow_checks=flows,scope='只有6个开度采样点改变；30分钟输出检查不能替代内部每步收敛验收')
(O/'首批修补运行核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));ss=pd.DataFrame(stats);print(ss[(ss.period=='全30天')&(ss.basis=='作者已核实修正')].to_string(index=False));print(json.dumps(manifest,ensure_ascii=False,indent=2))
lines=['# 首批输入修补30天验证结果','','C03的6点修补候选已完成原Pipedream同参数30天回放。原始观测和评分样本均不改；初始水位、流量与B00最大差小于1e-10。','','## 精度对照','','以下均排除t=0。作者88侧使用此前已核实的3条观测偏移，其余采用原始观测层，未删除本轮发现的评分水位尖点。','','| 范围 | B00水位MAE(m) | C03水位MAE(m) | 候选减基线(m) |','|---|---:|---:|---:|']
for area in ['全线118侧','作者88侧','七里河两侧','济河两侧']:
 x=ss[(ss.period=='全30天')&(ss.area==area)&(ss.basis=='作者已核实修正')].set_index('run');v0=x.loc[baseline,'mae'];v1=x.loc[candidate,'mae'];lines.append(f'| {area} | {v0:.6f} | {v1:.6f} | {v1-v0:+.6f} |')
lines+=['','## 输出异常检查','','| 版本 | 负流量站时次 | 最小全站流量(m³/s) | 最大全站流量(m³/s) |','|---|---:|---:|---:|']
for f in flows:lines.append(f"| {f['run']} | {f['negative_station_frames']} | {f['min_station_flow']:.3f} | {f['max_station_flow']:.3f} |")
lines+=['','这些是30分钟输出样本，不代表每个内部15秒步都已检查。原求解器默认一次求解，仍存在此前定位的分支及收敛问题，不能因进程完成就判定数值稳定。','','## 结论边界','','本轮已完成清单交付、6点候选补齐、输入差分检查和完整30天初步对照。只处理证据相对明确的短脉冲，不意味着所有输入问题已处理。109个模型内待核事件没有被判为正常；长时段目标/实际矛盾仍待逐段核实。即使局部误差改善，也不能据此证明补值就是现场真值。','','C02只生成输入并完成静态检查，尚未独立运行30天。C03不替换原C00基线，原始数据和冻结结果保持可回退。']
(O/'首批修补验证结果.md').write_text('\n'.join(lines)+'\n')
