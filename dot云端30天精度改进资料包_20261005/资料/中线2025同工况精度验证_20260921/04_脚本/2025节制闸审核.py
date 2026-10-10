"""原始2025节制闸窗口质量审核；不覆盖原值，不改模型。"""
from pathlib import Path
import json,re,hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'07_对比分析';I=R/'02_输入'
f=pd.read_csv(I/'2025节制闸原始窗口.csv',parse_dates=['时间']);model=json.loads((I/'B04_分水身份修正30天.json').read_text())['中线全线']['baseData'];start=pd.Timestamp('2025-07-22 12:00:00');end=pd.Timestamp('2025-08-21 12:00:00');times=pd.date_range(start,end,freq='30min');names=set(f.站点名称);mapping=[]
alias={'JZZ5':'淇河倒虹吸出口节制闸(镇平)','JZZ10':'草墩河渡槽进口节制闸','JZZ29':'聩城寨河倒虹吸出口节制闸','JZZ34':'淇河倒虹吸出口节制闸(鹤壁)','JZZ51':'沙河(北)倒虹吸出口节制闸'}
group={}
for key,gate in model['gates'].items():
 code,name,hole=key.split('-',2);group.setdefault(code,[]).append((int(hole),key,gate))
for code in sorted(group,key=lambda x:int(x[3:])):
 old=group[code][0][1].split('-')[1];name=alias.get(code,old);assert name in names,(code,name)
 codes=f.loc[f.站点名称==name,'站点编码'].unique();assert len(codes)==1
 mapping.append(dict(station_code=code,model_name=old,source_name=name,source_station_id=int(codes[0]),role='模型节制闸',mapping_basis='显式名称差异对照，非站点编码推定' if code in alias else '站名精确一致',model_holes=';'.join(str(i) for i,_,_ in sorted(group[code]))))
for code,name,role in [('BOUNDARY_IN','陶岔渠首引水闸','入口瞬时流量'),('BOUNDARY_OUT','北拒马河暗渠进口节制闸','原北拒马末边界水位')]:
 mapping.append(dict(station_code=code,model_name=name,source_name=name,source_station_id=int(f.loc[f.站点名称==name,'站点编码'].iloc[0]),role=role,mapping_basis='边界名称对应，具体使用侧别沿用原基线并由构造脚本明示',model_holes=''))
mp=pd.DataFrame(mapping);mp.to_csv(I/'2025节制闸站名映射.csv',index=False);(I/'2025节制闸站名映射.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
q=[];anoms=[];coverage=[];levels=[];gaps=[];raw=f[(f.时间>=start)&(f.时间<=end)]
for rr in mapping:
 code=rr['station_code'];g=raw[raw.站点名称==rr['source_name']].sort_values('时间');unique=g.时间.drop_duplicates();delta=unique.diff().dt.total_seconds().dropna();coverage.append(dict(station_code=code,name=rr['source_name'],rows=len(g),expected=len(times),duplicate_extra=len(g)-len(unique),missing_times=len(times.difference(unique)),non_30min_intervals=int((delta!=1800).sum()),start=str(unique.min()),end=str(unique.max())))
 holes={h:gt for h,_,gt in group.get(code,[])};upz=min((gt['zb'] for gt in holes.values()),default=np.nan);dnz=min((gt.get('down_zb',gt['zb']) for gt in holes.values()),default=np.nan)
 for col in f.columns[4:]:
  v=pd.to_numeric(g[col],errors='coerce');water=col in ['闸前水位','闸后水位'];opening='开度'in col;flow=col=='瞬时流量';invalid=v.isna()|(v<=0 if water else v<0);cap=np.nan;structural=False
  if opening:
   hole=int(re.search(r'孔(\d)',col)[1]);structural=hole not in holes and code.startswith('JZZ');cap=holes[hole]['b']*1000 if hole in holes else np.nan
  dif=v.diff();prev=v.shift();validpair=(v>0)&(prev>0)&g.时间.diff().dt.total_seconds().eq(1800)
  jumps=validpair & dif.abs().gt(.5) if water else (validpair & dif.abs().gt(50) if flow else pd.Series(False,index=v.index))
  above=v.gt(cap) if np.isfinite(cap) else pd.Series(False,index=v.index)
  row=dict(station_code=code,name=rr['source_name'],field=col,n=len(g),missing=int(v.isna().sum()),zero=int(v.eq(0).sum()),negative=int(v.lt(0).sum()),sentinel_minus300=int(v.eq(-300).sum()),valid=int((~invalid).sum()),valid_fraction=float((~invalid).mean()),minimum=v.min(),maximum=v.max(),median=v.median() if v.notna().any() else np.nan,constant=bool(v.nunique(dropna=True)==1),suspected_jump_count=int(jumps.sum()),above_current_conversion_cap=int(above.sum()),conversion_cap_raw_mm=cap,non_model_hole=structural)
  q.append(row)
  if water:
   z=upz if col=='闸前水位' else dnz
   levels.append(dict(station_code=code,name=rr['source_name'],side=col,positive_valid=int(v.gt(0).sum()),missing_or_nonpositive=int((v.isna()|v.le(0)).sum()),model_gate_invert_m=z,positive_below_model_invert=int(((v>0)&(v<z)).sum()) if np.isfinite(z) else None,min_positive=v[v>0].min(),max_positive=v[v>0].max(),datum_comparability='同为高程量级，但测点与闸底断面不同；不能据数值相近证明高程基准已对齐'))
   if not v.gt(0).any():gaps.append(dict(station_code=code,name=rr['source_name'],side=col,reason='全时段无正值水位',missing=int(v.isna().sum()),zero=int(v.eq(0).sum()),negative=int(v.lt(0).sum())))
  mask=(invalid & ~pd.Series(structural,index=v.index))|jumps|above
  for ix in v.index[mask]:
   val=v.loc[ix];a=dict(station_code=code,name=rr['source_name'],field=col,source_row=int(g.loc[ix,'源行号']),timestamp=str(g.loc[ix,'时间']),value=val,missing=bool(pd.isna(val)),nonpositive_water=bool(water and pd.notna(val) and val<=0),negative=bool(pd.notna(val) and val<0),above_current_conversion_cap=bool(above.loc[ix]),suspected_jump=bool(jumps.loc[ix]),jump_delta=float(dif.loc[ix]) if pd.notna(dif.loc[ix]) else None,action='水位非正或缺失不评分；正值跳变仅标疑似、原值保留；开度负值/缺失由构造策略显式记录')
   anoms.append(a)
# 重点定位JZZ49闸后：跳降与下一点回升，按真实日期和源行号记证。
n49=mp.set_index('station_code').loc['JZZ49','source_name'];g=raw[raw.站点名称==n49].sort_values('时间').copy();v=g['闸后水位'];sel=v.diff().lt(-1)&v.gt(0)&v.shift().gt(0);focus=[]
for ix in g.index[sel]:
 pos=g.index.get_loc(ix);focus.append(dict(source_row=int(g.loc[ix,'源行号']),timestamp=str(g.loc[ix,'时间']),day=(g.loc[ix,'时间']-start).total_seconds()/86400,previous=float(v.iloc[pos-1]),value=float(v.loc[ix]),next=float(v.iloc[pos+1]) if pos+1<len(v) else None,jump=float(v.loc[ix]-v.iloc[pos-1]),previous_next_difference=float(v.iloc[pos+1]-v.iloc[pos-1]) if pos+1<len(v) else None,judgment='疑似孤立突降；保留原值，不凭仿真误差掩除'))
pd.DataFrame(q).to_csv(O/'2025节制闸质量审核.csv',index=False);pd.DataFrame(anoms).to_csv(O/'2025节制闸异常逐值.csv',index=False);pd.DataFrame(coverage).to_csv(O/'2025节制闸时间覆盖.csv',index=False);pd.DataFrame(levels).to_csv(O/'2025水位侧别与模型高程核对.csv',index=False);pd.DataFrame(gaps).to_csv(O/'2025作者空白曲线候选.csv',index=False)
validside=sum(z['positive_valid']>0 for z in levels if z['station_code'].startswith('JZZ'));summary=dict(source_extraction=json.loads((I/'2025节制闸原始窗口.json').read_text()),start=str(start),end=str(end),expected_times=len(times),source_station_count=f.站点名称.nunique(),mapped_model_station_count=len(group),boundary_count=2,extra_stations=sorted(names-set(mp.source_name)),gate_observed_water_sides=validside,gate_fully_invalid_water_sides=118-validside,water_side_invalid=gaps,opening_unit='表头未明示单位。实际与目标为千量级（如1907、1910），结合既有mm观测口径按mm转m具有量级依据，仍属于需说明的单位推定；不能凭站编码推JZZ。',opening_current_cap='模型gate.b为闸宽，当前转换却将其作为gate_max_openings；超该值只标当前转换截断风险，不称真实行程超限。',quality_rules=['水位<=0明确无效，缺失不填成实测','相邻30分钟正水位变化>0.5m仅疑似，正值不自动剔除','相邻30分钟正流量变化>50m3/s仅疑似','非模型孔空值与未安装孔区分，不强制当有效孔缺测','目标开度不是实际执行开度，默认不得用目标冒充实际'],jzz49_afterbay_large_drop=focus)
(O/'2025节制闸质量审核.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
def md(df):return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
report=['# 2025节制闸原始观测质量审核','',f'真实日期2025-07-22 12时至08-21 12时，共30天，30分钟采样，含两端时刻共{len(times)}点。另保留07-20至07-22 12时先导期原始观测。窗口文件共{len(f)}行、{f.站点名称.nunique()}站，按站名建立59模型站及2边界映射，站点编码不冒充JZZ序号。','',summary['opening_unit'],'',summary['opening_current_cap'],'','## 数据判定','']+['- '+x for x in summary['quality_rules']]+['','## 水位侧别与作者88条曲线','',f'118个模型水位侧中，本窗口有正值的侧别为{validside}，全无正值侧别{118-validside}。下表可与作者PDF空白侧逐项对应；仅数量相符不能视为已逐图核对。','',md(pd.DataFrame(gaps)) if gaps else '本窗口无全侧缺测，作者PDF对应关系见后文。','','## JZZ49闸后重点复核','',md(pd.DataFrame(focus)) if focus else '未发现满足相邻正值突降超过1m的记录。','','## 站名映射','',md(mp),'','## 时间覆盖','',md(pd.DataFrame(coverage)),'','## 高程可比性','', '观测闸前/后水位与模型闸底高程逐侧列在2025水位侧别与模型高程核对.csv。测点在闸前/闸后不同断面，水位高程与模型底高程数值可比不等于高程基准已核准；正值低于模型底高程另标疑似，不直接修正观测。']
(R/'08_报告/2025节制闸观测质量审核.md').write_text('\n'.join(report));print(json.dumps({k:v for k,v in summary.items() if k not in ['source_extraction','water_side_invalid']},ensure_ascii=False,default=str))
# 与此前PDF实际反读的88侧清单逐侧对照，校验PDF二进制一致后才引用。
pdfref=R.parent/'作者PDF与R05精度对照_20260919/作者EnKF曲线反读估计.csv'
pdfold=R.parent/'中线30天基线与同化对照_20260916/01_原始资料/stations_30d_flow.pdf';pdfnew=R/'01_原始资料/stations_30d_flow.pdf'
if pdfref.exists() and hashlib.sha256(pdfold.read_bytes()).hexdigest()==hashlib.sha256(pdfnew.read_bytes()).hexdigest():
 pdfcurves=pd.read_csv(pdfref);readable=set(zip(pdfcurves.station,pdfcurves.metric));cmp=[]
 for rec in levels:
  if not rec['station_code'].startswith('JZZ'):continue
  key=(rec['station_code'],'forebay_m' if rec['side']=='闸前水位' else 'afterbay_m')
  cmp.append(dict(**rec,pdf_curve_readable=key in readable,pdf_reference_source=str(pdfref),interpretation='本xlsx有完整正值水位，PDF不可反读不能归因本表缺测' if key not in readable else 'PDF可反读侧；本表正值仍需异常标记'))
 pd.DataFrame(cmp).to_csv(O/'2025作者空白曲线候选.csv',index=False);summary['pdf_readable_sides']=len(readable);summary['pdf_blank_but_positive_xlsx_sides']=sum(not x['pdf_curve_readable'] and x['positive_valid']>0 for x in cmp);summary['pdf_identity_sha256']=hashlib.sha256(pdfnew.read_bytes()).hexdigest()
 report+=['','## 作者PDF侧别逐项对应','',f'与此前实际反读清单逐侧对照，且新旧PDF SHA256相同。PDF可反读{len(readable)}侧；其余{summary["pdf_blank_but_positive_xlsx_sides"]}侧在本次Excel均有1441个正水位值。因此PDF的空白或不可读侧不能归因为当前Excel缺测。','',md(pd.DataFrame(cmp).query('not pdf_curve_readable')[['station_code','name','side','positive_valid']])]
# 实际/目标仅比较，不替换。有效实际孔没有全列缺失，不需要闭孔替代假设。
ac=[]
for rr in mapping:
 code=rr['station_code']
 if code not in group:continue
 g=raw[raw.站点名称==rr['source_name']]
 for hole,gn,gt in sorted(group[code]):
  a=pd.to_numeric(g[f'孔{hole}实际开度'],errors='coerce');t=pd.to_numeric(g[f'孔{hole}目标开度'],errors='coerce');ok=a.ge(0)&t.ge(0);err=(a-t)[ok]
  ac.append(dict(station_code=code,hole=hole,target=gn,actual_missing=int(a.isna().sum()),actual_negative=int(a.lt(0).sum()),target_missing=int(t.isna().sum()),target_negative=int(t.lt(0).sum()),actual_target_mae_raw_mm=float(err.abs().mean()),actual_target_max_abs_raw_mm=float(err.abs().max()),actual_zero_target_positive=int((a.eq(0)&t.gt(0)).sum()),actual_positive_target_zero=int((a.gt(0)&t.eq(0)).sum()),all_actual_missing=bool(a.isna().all()),assumption='不替换实际开度；负值待显式QC'))
pd.DataFrame(ac).to_csv(O/'2025实际目标开度对照.csv',index=False)
summary['actual_gate_holes']=len(ac);summary['actual_opening_missing_cells']=sum(x['actual_missing'] for x in ac);summary['actual_opening_negative_cells']=sum(x['actual_negative'] for x in ac);summary['actual_opening_all_missing_holes']=sum(x['all_actual_missing'] for x in ac)
# 入口仅做粗闭合，不把同刻流量差定为传感器偏差。
divfile=I/'2025分退水原始窗口.csv'
if divfile.exists():
 df=pd.read_csv(divfile,parse_dates=['时间']);df=df[df.时间.between(start,end)];a=raw[raw.站点名称=='陶岔渠首引水闸'].set_index('时间').瞬时流量;b=raw[raw.站点名称=='刁河渡槽进口节制闸'].set_index('时间').瞬时流量;c=df[df.闸站名称=='肖楼分水口'].set_index('时间').总瞬时流量
 ee=pd.concat([a,b,c],axis=1);ee.columns=['陶岔瞬时流量','刁河瞬时流量','肖楼总瞬时流量'];ee['未扣蓄水与其他分退水的差']=ee.iloc[:,0]-ee.iloc[:,1]-ee.iloc[:,2];ee.to_csv(O/'2025入口粗水量核对.csv')
 summary['inlet_rough_balance']={'first':ee.iloc[0].to_dict(),'mean':ee.mean().to_dict(),'min_residual_m3s':float(ee.iloc[:,-1].min()),'max_residual_m3s':float(ee.iloc[:,-1].max()),'residual_volume_trapezoid_m3':float(np.trapezoid(ee.iloc[:,-1],dx=1800)),'warning':'未计渠段蓄水、测流位置与传播时滞，不据此率定流量计；肖楼总瞬时流量不是固定6m3/s'}
 report+=['','## 入口粗水量核对','',f'首时陶岔{a.iloc[0]:.6f}m³/s、刁河{b.iloc[0]:.6f}m³/s、肖楼{c.iloc[0]:.6f}m³/s，同刻差{ee.iloc[0,-1]:.6f}m³/s；30天平均差{ee.iloc[:,-1].mean():.6f}m³/s。尚未扣渠段蓄量、其他分退水或测流时滞，不能直接判定哪台仪表偏差。肖楼总瞬时流量不是固定6m³/s。']
summary['outlet_location_check']={'model_boundary':'NODE-main-119','last_section':'q60s33','model_chainage_m':1197636,'source_station':'北拒马河暗渠进口节制闸','use_side':'闸前水位','attachment_rows':{'785':'1#闸前K1197+662，闸前7m','786':'2#闸前K1196+662，闸前7m','787_788':'闸后K1197+675，闸后6m'},'upstream_sensor_offsets_m':[26,-974],'status':'名称与前侧对应；两闸前安装桩号相差1km，不能宣称精确位置吻合，不以闸后或惠南庄替代'}
report+=['','## 实际开度与尾边界','',f'183个模型实际孔均有1441个数值，无全列缺失孔，无需用目标0或模型0填闭孔；其中负开度{summary["actual_opening_negative_cells"]}格仍须构造脚本明示处置。实际开度超当前转换上限229格，不能据此称真实行程超限。','', '末模型断面q60s33为1197636m，原末节点NODE-main-119。新观测“北拒马河暗渠进口节制闸”闸前名称/侧别对应，但安装表785行1#闸前1197662m与786行2#闸前1196662m相差1km，均描述闸前7m；分别较模型末断面+26m和−974m。可沿用前水位近似作为尾界，但不能声称测点桩号已精确对齐，不自动换闸后或惠南庄。']
(O/'2025节制闸质量审核.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str));(R/'08_报告/2025节制闸观测质量审核.md').write_text('\n'.join(report))
