from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parent;B=R/'26_冻结基线与后续归因';manifest=json.loads((B/'冻结文件校验.json').read_text());rows=[];details=[];sources=[]
for label,folder,phase in [('B01改水位同化','B01_允许修改水位的状态同化参考基线','analysis_after_update'),('B02改参数同化','B02_渠池归属修复后的参数同化诊断基线','forecast_before_update')]:
 data={}
 for role in ['同化运行','匹配无同化对照']:
  p=B/folder/role/'预报与分析逐点.csv';sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==manifest[str(p.relative_to(B))]['sha256'];sources.append(dict(source=str(p),sha256=sha));d=pd.read_csv(p);d=d[(d.operator=='闸端全线')&(d.phase==(phase if role=='同化运行' else 'forecast_before_update'))].copy();d=d[(d.time_s>0)&(d.time_s<=21600)].copy();d['error']=d.sim-d.observed;assert np.isfinite(d[['sim','observed']]).all().all();assert not d.duplicated(['station','metric','time_s']).any();data[role]=d
 a=data['同化运行'];b=data['匹配无同化对照'];key=['station','metric','time_s'];a=a.sort_values(key);b=b.sort_values(key);assert np.array_equal(a[key],b[key]) and np.array_equal(a.observed,b.observed)
 for metric,title in [('forebay_m','闸前'),('afterbay_m','闸后')]:
  g=a[a.metric==metric];m=b[b.metric==metric];assert len(g)==len(m)==59*12
  for st,ss in g.groupby('station'):assert np.array_equal(np.sort(ss.time_s),np.arange(1,13)*1800)
  means=g.groupby('station').error.apply(lambda x:abs(x).mean());maxima=g.groupby('station').error.apply(lambda x:abs(x).max());worst=g.loc[g.error.abs().idxmax()]
  rows.append(dict(baseline=label,position=title,n=len(g),stations=59,model_mae=abs(m.error).mean(),assimilation_mae=abs(g.error).mean(),assimilation_rmse=np.sqrt((g.error**2).mean()),assimilation_max_abs_error=abs(g.error).max(),station_mae_le_0p1=int((means<=.1).sum()),station_all_times_le_0p1=int((maxima<=.1).sum()),sample_le_0p1_percent=100*float((g.error.abs()<=.1).mean()),mae_improvement_percent=100*(1-abs(g.error).mean()/abs(m.error).mean()),worst_station=worst.station,worst_time_s=float(worst.time_s)))
  for st,z in g.groupby('station'):details.append(dict(baseline=label,station=st,position=title,mae=abs(z.error).mean(),rmse=np.sqrt((z.error**2).mean()),max_abs_error=abs(z.error).max()))
s=pd.DataFrame(rows);s.to_csv(O/'首六小时闸前闸后总体精度.csv',index=False);pd.DataFrame(details).to_csv(O/'首六小时逐闸精度明细.csv',index=False)
name_map={q['station_code']:q['source_name'] for q in json.loads((R/'02_输入/2025节制闸站名映射.json').read_text())};dt=pd.DataFrame(details);tab=dt.pivot(index='station',columns=['baseline','position'],values='mae');lines=['# 两套冻结基线闸前、闸后水位精度','','统计范围为59座闸、起始6小时（2025-07-22 12:00至18:00），每个测点12个半小时时刻，不含t=0；闸前和闸后分别有708个误差样本。这6小时内持续同化，不是停止接收新实测后的6小时独立预报。单位m。与28目录最新三线图同口径：B01为当前实测修正后的分析水位，B02为参数更新后继续计算、尚未消费当前实测的水位。实测采用既定已核实修正层，保留既定有效观测，不另删尖点。','','MAE = 平均值(|同化水位−实测水位|)；无同化模型MAE另按模型水位−实测水位计算。MAE是绝对差平均，不是同化与无同化两条线之间的差。','','## 总体统计','','|基线|位置|无同化模型MAE|同化MAE|同化RMSE|同化最大绝对误差|测点窗口MAE≤0.1m|测点全时刻误差≤0.1m|单次误差≤0.1m比例|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['baseline']}|{r['position']}|{r['model_mae']:.4f}|{r['assimilation_mae']:.4f}|{r['assimilation_rmse']:.4f}|{r['assimilation_max_abs_error']:.4f}|{r['station_mae_le_0p1']}/59|{r['station_all_times_le_0p1']}/59|{r['sample_le_0p1_percent']:.1f}%|")
lines+=['','两类橙线使用当前观测的方式不同，不作为纯算法排名。历史边界驱动回放也不是未知未来输入的独立预报。测点窗口MAE≤0.1m不能等同该测点每个时刻都≤0.1m。最大误差按现有全部有效观测统计；不据误差大单独判传感器异常。','','## 逐闸首6小时MAE','','|节制闸|改水位同化：闸前|改水位同化：闸后|改参数同化：闸前|改参数同化：闸后|','|---|---:|---:|---:|---:|']
for st in sorted(tab.index,key=lambda k:int(k[3:])):lines.append('|'+st+' '+name_map.get(st,st)+'|'+'|'.join(f'{tab.loc[st,(b,p)]:.4f}' for b in ['B01改水位同化','B02改参数同化'] for p in ['闸前','闸后'])+'|')
(O/'首六小时两基线闸前闸后精度表.md').write_text('\n'.join(lines)+'\n');(O/'首六小时统计来源核验.json').write_text(json.dumps(dict(sources=sources,phases={'B01': 'analysis_after_update','B02':'forecast_before_update'},rows=rows),ensure_ascii=False,indent=2));print(s.to_string(index=False))
