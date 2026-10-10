from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
O=Path(__file__).resolve().parent;R=O.parent;I=R/'07_对比分析';p=O/'两基线闸前闸后精度表.md';original=p.read_text();marker='## 王孝群PDF的30天MAE粗估参考'
base=original.split('\n'+marker)[0].rstrip();rows=[];bykind={};sources=[]
for kind,file in [('WXQ确定性模型','作者PDF确定性误差同网格估计.csv'),('WXQ图示EnKF同化','作者PDF同网格误差估计.csv')]:
 path=I/file;d=pd.read_csv(path);d=d[(d.time_s>0)&(d.time_s<=2592000)].copy();assert d.covered.all() and np.isfinite(d.error_estimate).all() and not d.duplicated(['station_code','metric','time_s']).any();assert len(d)==88*1440
 for keys,g in d.groupby(['station_code','metric']):assert np.array_equal(g.time_s.to_numpy(),np.arange(1,1441)*1800)
 bykind[kind]=d.groupby(['station_code','metric']).error_estimate.apply(lambda x:abs(x).mean())
 for metric,title in [('forebay_m','闸前'),('afterbay_m','闸后'),(None,'合并')]:
  g=d if metric is None else d[d.metric==metric];means=g.groupby(['station_code','metric']).error_estimate.apply(lambda x:abs(x).mean());rows.append(dict(source=kind,position=title,points=len(means),samples=len(g),mae_estimate=abs(g.error_estimate).mean(),rmse_estimate=np.sqrt((g.error_estimate**2).mean()),max_abs_error_estimate=abs(g.error_estimate).max(),point_mae_gt_0p1=int((means>.1).sum())))
 sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert bykind['WXQ确定性模型'].index.equals(bykind['WXQ图示EnKF同化'].index)
a=pd.DataFrame(rows);a.to_csv(O/'王孝群PDF闸前闸后粗估.csv',index=False)
ours=pd.read_csv(O/'逐闸精度明细.csv');ours['metric']=ours.position.map({'闸前':'forebay_m','闸后':'afterbay_m'});common=set(bykind['WXQ图示EnKF同化'].index);ours=ours[[ (r.station,r.metric) in common for r in ours.itertuples()]];comparison=[]
for label,g in ours.groupby('baseline'):
 assert len(g)==88
 comparison.append(dict(source=label,front=g[g.metric=='forebay_m'].mae.mean(),back=g[g.metric=='afterbay_m'].mae.mean(),combined=g.mae.mean()))
name_map={r['station_code']:r['source_name'] for r in json.loads((R/'02_输入/2025节制闸站名映射.json').read_text())}
lines=['',marker,'','**粗估结论：王孝群PDF中的确定性模型MAE约0.048m（4.8cm），图示EnKF同化MAE约0.030m（3.0cm）。** 这是PDF矢量误差曲线读取、按半小时网格线性插值后的估算，不是作者原始数值输出。','', '|评价范围|可识别水位测点数|WXQ确定性模型MAE粗估|WXQ图示EnKF同化MAE粗估|','|---|---:|---:|---:|']
for position in ['闸前','闸后','合并']:
 r=a[(a.source=='WXQ确定性模型')&(a.position==position)].iloc[0];q=a[(a.source=='WXQ图示EnKF同化')&(a.position==position)].iloc[0];lines.append(f'|{position}|{q.points}|约{r.mae_estimate:.3f}m|约{q.mae_estimate:.3f}m|')
lines+=['','图册覆盖59座闸，但只有88个测点的误差曲线能按既定规则可靠读取，其中闸前43个、闸后45个。其他曲线缺失/不可用的测点不填0，也不宣称其精度已达标。','本次与本MD一致，使用2025-07-22 12:00至2025-08-21 12:00的(t0,t0+30天]1440个半小时时刻，不含起始点；旧报告含起始点1441个时刻的整体估计为0.04777164m、0.02980033m，本次重新汇总分别为0.04779505m、0.02981127m。差别来自是否计入起始点，不是曲线或原数据改变。','','### 同一批88个测点的数值参考','','以下仅统一了测点集合和时间网格，并未统一模型、输入处理、同化对象或使用当前观测的方式，不能据此宣布算法优劣。WXQ图例为“流量同化EnKF”，其具体同化变量、初始化、重新初始化频次、参数与先验/分析输出口径仍未获得原程序确认。','','|结果来源|闸前43个测点MAE|闸后45个测点MAE|合并88个测点MAE|','|---|---:|---:|---:|','|WXQ确定性模型（图上估计）|约0.049m|约0.047m|约0.048m|','|WXQ图示EnKF（图上估计）|约0.030m|约0.029m|约0.030m|']
for r in comparison:lines.append(f"|{r['source']}（原始结果统计）|{r['front']:.4f}m|{r['back']:.4f}m|{r['combined']:.4f}m|")
lines+=['','### WXQ是否仍有30天MAE大于0.1m的测点？','','在可识别的88个测点中，图示EnKF的**估计月MAE均不超过0.1m**，确定性模型则有12个超过0.1m（闸前7个、闸后5个）。这不等于作者全部118个测点都通过，也不等于每一时刻误差≤0.1m；图上同化最大瞬时绝对误差仍约4.97m，保留该值，不用月平均掩盖尖峰。','','### WXQ逐闸MAE粗估','','单位m。“未估”表示没有可用误差曲线，不表示0。此表保留与前文相同的59座闸，便于查找仍有偏差的对应位置。','','|节制闸|WXQ确定性闸前|WXQ同化闸前|WXQ确定性闸后|WXQ同化闸后|','|---|---:|---:|---:|---:|']
for st in sorted(name_map,key=lambda x:int(x[3:]) if x.startswith('JZZ') else 999):
 if not st.startswith('JZZ') or not 2<=int(st[3:])<=60:continue
 vals=[]
 for met in ['forebay_m','afterbay_m']:
  for kind in ['WXQ确定性模型','WXQ图示EnKF同化']:
   value=bykind[kind].get((st,met),None);vals.append('未估' if value is None else f'约{value:.3f}')
 lines.append('|'+st+' '+name_map[st]+'|'+'|'.join(vals)+'|')
lines+=['','来源：01_原始资料/stations_30d_flow.pdf；07_对比分析/作者PDF同网格误差估计审核.json、作者PDF确定性提取审核.json及对应网格估算CSV。读取的是图中“模型−实测”误差曲线，MAE为曲线纵值绝对值的时间平均；曲线简化与插值带来估算误差。']
new=base+'\n'+'\n'.join(lines)+'\n';p.write_text(new);assert p.read_text().startswith(base)
(O/'王孝群PDF参考来源核验.json').write_text(json.dumps(dict(pdf=str((R/'01_原始资料/stations_30d_flow.pdf').resolve()),pdf_sha256=hashlib.sha256((R/'01_原始资料/stations_30d_flow.pdf').read_bytes()).hexdigest(),sources=sources,points=88,t0_excluded=True,points_per_series=1440,rows=rows,matched_baseline_comparison=comparison,method='PDF简化矢量误差曲线插值估计，不是作者原始输出'),ensure_ascii=False,indent=2));print(a.to_string(index=False));print(comparison)
