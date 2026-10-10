from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;P=O.parent
names=['原几何固定参数','开敞几何外延假设']
old=pd.read_csv(P/'两分支水位与实测.csv')
results=[];scores=[];checks=[];audits=[]
for name in names:
 d=O/name;r=json.loads((d/'三十天单分支结果.json').read_text());progress=json.loads((d/'进度.json').read_text());a=pd.read_csv(d/'水位与实测.csv.gz');v=pd.read_csv(d/'几何与储水逐半小时.csv')
 assert r['status']==progress['status']=='COMPLETED'
 assert r['runs'][0]['cycles']==1428 and len(a)==1428*118 and r['runs'][0]['parameters_unchanged']
 assert np.array_equal(np.sort(a.time_s.unique()),np.arange(23400,2592001,1800))
 assert a.groupby(['station_code','metric']).size().eq(1428).all() and np.isfinite(a.sim).all()
 assert r['within_original_exact_all'] and r['runs'][0]['highest_open_mean_depth_m']<99 and not r['runs'][0]['numba']['jit_disabled']
 b=old[old['case']==name].merge(a[a.time_s<=259200],on=['time_s','station_code','metric'],validate='one_to_one',suffixes=('_3day','_long'))
 assert len(b)==132*118
 diff=float(abs(b.sim_3day-b.sim_long).max());assert diff==0
 checks.append(dict(case=name,first_132_cycles_matched_values=len(b),max_abs_water_difference_m=diff,within_original_geometry_identical=r['within_original_exact_all'],parameter_unchanged=r['runs'][0]['parameters_unchanged']))
 results.append(r);scores.append(a);audits.append(v)
summary=dict(status='COMPLETED',start_s=21600,end_s=2592000,periods_per_case=1428,not_complete_1440_cycle_run=True,no_assimilation=True,checks=checks,metrics=[m for r in results for m in r['metrics']],runs=[r['runs'][0] for r in results],initial_volume_difference_m3=results[1]['runs'][0]['initial_full_nonfixed_volume_m3']-results[0]['runs'][0]['initial_full_nonfixed_volume_m3'])
assert summary['initial_volume_difference_m3']==json.loads((P/'三天断面外延假设结果.json').read_text())['initial_geometry_volume_difference_m3']
rows=[]
for window,lo in [('第6小时至第30天',21600),('第3天后至第30天',259200)]:
 parts=[]
 for name,a in zip(names,scores):
  a=a[a.time_s>lo];g=a.groupby(['station_code','metric']).error.agg(mae=lambda x:abs(x).mean(),max_abs=lambda x:abs(x).max()).reset_index();g['case']=name;parts.append(g)
 t=pd.concat(parts).pivot(index=['station_code','metric'],columns='case',values=['mae','max_abs']);t.columns=['_'.join(q) for q in t.columns];t=t.reset_index();t['window']=window;t['mae_change_m']=t['mae_开敞几何外延假设']-t['mae_原几何固定参数'];rows.extend(t.to_dict('records'))
pd.DataFrame(rows).to_csv(O/'逐水位测点改善恶化.csv',index=False)
summary['point_changes']={}
for w in ['第6小时至第30天','第3天后至第30天']:
 t=pd.DataFrame([z for z in rows if z['window']==w]);summary['point_changes'][w]=dict(improved=int((t.mae_change_m<-1e-6).sum()),worsened=int((t.mae_change_m>1e-6).sum()),within_1micrometer=int((t.mae_change_m.abs()<=1e-6).sum()),improved_more_than_1cm=int((t.mae_change_m<-.01).sum()),worsened_more_than_1cm=int((t.mae_change_m>.01).sum()))
a=scores[0].merge(scores[1],on=['time_s','station_code','metric'],validate='one_to_one',suffixes=('_old','_ext'));a['trajectory_difference_m']=a.sim_ext-a.sim_old;worst=a.loc[a.trajectory_difference_m.abs().idxmax()];summary['max_trajectory_difference']=dict(station=worst.station_code,metric=worst.metric,time_s=float(worst.time_s),signed_m=float(worst.trajectory_difference_m))
(O/'三十天配对核验与统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False));pd.DataFrame(summary['metrics']).to_csv(O/'两分支误差指标.csv',index=False)
lines=['# 开敞断面外延假设：第6小时至第30天配对报告','',
'## 实验边界','',
'这是条件性的数学外延总效应试验，不是工程修复。相同B02第6小时成员0的H/h/Q、原119参数和历史边界，不再同化；只在内存中将2415条开敞连接的几何上限改为固定100m，原竖墙/边坡继续向上延伸。原轮廓内A/P/B逐点完全相同；105倒虹吸、所有闸门、底高程、糙率、闸流系数不改。','',
'本次起点21600s、终点2592000s，每分支1428个半小时周期，覆盖29.75天；不是从0秒起算的完整1440周期。后27天口径为t>259200s至第30天。三天原产物保留不改。','',
'两分支从同一6小时检查点重跑，前三天各15576个水位结果与原三天试验逐值一致。原几何三天结果此前也已与33/L00对照完全一致，因此几何/储水审计本身没有改变推进。','',
'## MAE与最大误差','',
'单位m，与相同既定有效实测比。','',
'|窗口|范围|原几何MAE|外延假设MAE|原几何最大误差|外延假设最大误差|','|---|---|---:|---:|---:|---:|']
for w in ['第6小时至第30天','第3天后至第30天','第6小时至第3天']:
 for scope in ['118个水位测点','作者88个可比水位测点']:
  p={z['case']:z for z in summary['metrics'] if z['window']==w and z['scope']==scope};b=p[names[0]];e=p[names[1]];lines.append(f"|{w}|{scope}|{b['mae_m']:.6f}|{e['mae_m']:.6f}|{b['max_abs_m']:.6f}|{e['max_abs_m']:.6f}|")
lines+=['','RMSE、最大误差对应测点和时刻完整保留于两分支误差指标.csv。','',
'## 逐测点变化','',
'|窗口|MAE减小测点|MAE增大测点|变化不超过1微米|改善超过1cm|恶化超过1cm|','|---|---:|---:|---:|---:|---:|']
for w,cnt in summary['point_changes'].items():lines.append(f"|{w}|{cnt['improved']}|{cnt['worsened']}|{cnt['within_1micrometer']}|{cnt['improved_more_than_1cm']}|{cnt['worsened_more_than_1cm']}|")
lines+=['','1微米只用于完整计数分类，不是工程显著性标准。逐点清单另表提供。','',
'|全窗口变化较大的测点|原几何MAE|外延MAE|变化m|','|---|---:|---:|---:|']
t=pd.DataFrame([z for z in rows if z['window']=='第6小时至第30天']).sort_values('mae_change_m')
for _,z in pd.concat([t.head(5),t.tail(5)]).iterrows():lines.append(f"|{z.station_code} {'闸前' if z.metric=='forebay_m' else '闸后'}|{z['mae_原几何固定参数']:.6f}|{z['mae_开敞几何外延假设']:.6f}|{z.mae_change_m:+.6f}|")
lines+=['','## 同水位下初始几何蓄量与水量口径','',f"初始H/h/Q完全相同，但外延使初始非定水位控制体几何蓄量增加 **{summary['initial_volume_difference_m3']:.3f}m³**。这个体积差不是新来水或同化补水，而是几何解释改变。因此本试验同时改变过水面积、湿周和储水表达，不能声称只测出A/B不一致的净贡献。",'',
'|第6小时至第30天累计量|原几何|外延假设|','|---|---:|---:|']
b,e=[z['last_audit'] for z in summary['runs']]
for key,label in [('open_geometric_minus_B_m3','开敞几何蓄量变化 − 累计B储水变化 m³'),('cumulative_abs_open_geometry_B_mismatch_m3','上述逐步差额绝对值累计 m³'),('net_minus_discrete_m3','全控制体净入流 − 离散储水变化 m³'),('net_minus_geometry_m3','全控制体净入流 − 几何储水变化 m³'),('original_contour_exceed_nodes','末时超过原轮廓的计算断面数量'),('original_cap_exceed_links','末时超过原g1的开敞连接数量')]:lines.append(f"|{label}|{b[key]:.6f}|{e[key]:.6f}|")
lines+=['','离散方程闭合不等于原始断面正确；A·L几何体积与B储水差还受有限时间步的非线性等影响。封闭结构的窄缝储水表达未修改。原轮廓仍被超过，外延不能证明实际现场允许这些水位，也不能据此宣称漫顶。','',
'## 数值核验','',
'|分支|正式周期|接受时间步|减步重试|最大水位迭代残差m|最大闸流残差m³/s|最高开敞平均水深m|','|---|---:|---:|---:|---:|---:|---:|']
for z in summary['runs']:lines.append(f"|{z['case']}|{z['cycles']}|{z['steps']}|{z['retries']}|{z['max_head_residual_m']:.8f}|{z['max_gate_residual_m3s']:.8f}|{z['highest_open_mean_depth_m']:.6f}|")
lines+=['','全程使用原15s步长、num_iter=20、strict=True、最小失败减步1.875s规则，未放宽门槛。每步六类水力状态有限，参数全程固定；Numba运行签名和模型身份均保留。100m数学上限未被触及。','',
'## 应如何使用结果','',
'先看配对MAE与逐点改善/恶化，再看几何储水一致性。不能因后一项变好就宣布与现场实测更吻合，也不能以局部改善推定全线获益。这里只给数学假设的总效应；工程实施仍需原断面与结构类型的独立依据。既有B01/B02基线未替换，历史边界回放也不是未知未来输入的独立预报。','',
'文件：三十天配对核验与统计.json、两分支误差指标.csv、逐水位测点改善恶化.csv；两分支目录分别保留压缩水位过程、储水逐周期审计、末态和Numba/严格门槛证据。']
lines[2:2]=['## 结论','',
'**延长到第30天后仍没有整体精度收益，不建议据此替换基线或直接外延工程断面。** 全窗口118测点MAE从0.251203增至0.253545m，作者88测点从0.236349增至0.238370m；36个测点改善、82个恶化，其中9个恶化超过1cm，没有测点改善超过1cm。最大误差略降，但不能抵消普遍MAE增大。','',
'开敞A·L与B储水逐步口径差的绝对值累计从约247859m³降至324m³；这说明数学一致性改善，并不证明现场水位更准。相同水位下初始几何蓄量增加297907m³，须与水位结果一起解释。','']
(O/'开敞断面外延假设三十天配对报告.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
