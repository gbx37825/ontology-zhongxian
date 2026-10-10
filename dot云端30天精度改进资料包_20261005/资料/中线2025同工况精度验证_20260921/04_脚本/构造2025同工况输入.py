from pathlib import Path
import copy,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
start=pd.Timestamp('2025-07-22 12:00:00');end=pd.Timestamp('2025-08-21 12:00:00');idx=pd.date_range('2025-07-20',end,freq='30min');formal=idx[(idx>=start)&(idx<=end)];times=np.arange(len(formal))*1800
G=pd.read_csv(R/'02_输入/2025节制闸原始窗口.csv',parse_dates=['时间']);O=pd.read_csv(R/'02_输入/2025分退水原始窗口.csv',parse_dates=['时间'])
GM=pd.read_csv(R/'02_输入/2025节制闸站名映射.csv').set_index('station_code');OM=json.loads((R/'02_输入/2025分退水映射.json').read_text())
base=json.loads((R/'02_输入/B04_分水身份修正30天.json').read_text());bm=base if 'baseData'in base else next(iter(base.values()));fixes=[];provenance=[]
def aligned(frame,name,col):
 g=frame[frame['站点名称' if '站点名称'in frame.columns else '闸站名称']==name].sort_values('时间');assert not g['时间'].duplicated().any(),name
 s=g.set_index('时间')[col].reindex(idx).astype(float);assert len(g)==len(idx),(name,len(g));return s

def fill_bad(s,bad,kind,name,allow_zero=False):
 v=s.copy();v[bad]=np.nan
 ix=np.flatnonzero(v.isna());segments=np.split(ix,np.where(np.diff(ix)>1)[0]+1) if len(ix) else []
 for seg in segments:
  a,b=int(seg[0]),int(seg[-1]);left=a-1;right=b+1
  zero=allow_zero and (v.dropna()==0).all()
  if zero:replacement=np.zeros(len(seg));reason='有效观测全为0的白庄，坏段按0补值假设'
  else:
   assert left>=0 and right<len(v) and pd.notna(v.iloc[left]) and pd.notna(v.iloc[right]),(name,'缺补值端点',idx[a],idx[b])
   span=(idx[right]-idx[left]).total_seconds();assert span<=7200,(name,'缺测跨度超过2h',span)
   replacement=np.interp(np.arange(a,b+1),[left,right],[v.iloc[left],v.iloc[right]]);reason='有效端点间不超过2h，按时间线性对齐；原值保留'
  for j,value in zip(seg,replacement):
   if start<=idx[j]<=end:fixes.append(dict(kind=kind,name=name,timestamp=str(idx[j]),original=None if pd.isna(s.iloc[j]) else float(s.iloc[j]),replacement=float(value),reason=reason))
   v.iloc[j]=value
 assert np.isfinite(v).all(),name
 return v.loc[formal].to_numpy()

# 只取总瞬时流量；128个主渠分水对象，支线串联记录不重复计入。
bleed={};used={}
for entry in OM:
 target=entry['target'];names=entry['source_names']
 if target.startswith('FSK121-'):names=['西黑山进口闸']
 assert names,(target,'缺少源站')
 value=np.zeros(len(formal))
 for name in names:
  raw=aligned(O,name,'总瞬时流量');bad=raw.isna()|(raw<=-100)
  value+=fill_bad(raw,bad,'分退水',name,allow_zero=(name=='白庄分水口'))
  used.setdefault(name,[]).append(target)
 bleed[target]=value
 provenance.append(dict(type='bleeders',target=target,source_names=names,source_column='总瞬时流量',note='西黑山XW支线入口别名假设；不重复扣文村北' if target.startswith('FSK121-') else entry.get('note','')))
assert all(len(set(v))==1 for v in used.values()),used
actual={};target_open={};limits={}
for name,gate in bm['baseData']['gates'].items():
 station=name.split('-')[0];hole=int(name.rsplit('-',1)[1]);source=GM.loc[station,'source_name']
 raw=aligned(G,source,f'孔{hole}实际开度');v=fill_bad(raw,raw.isna()|(raw<0),'实际开度',name)/1000
 tar=aligned(G,source,f'孔{hole}目标开度').loc[formal].to_numpy()/1000;assert np.isfinite(tar).all() and (tar>=0).all()
 actual[name]=v;target_open[name]=tar;limits[name]=float(max(gate['b'],max(v),max(tar)))
 provenance.append(dict(type='gates',target=name,source_names=[source],actual_column=f'孔{hole}实际开度',target_column=f'孔{hole}目标开度',units='原mm除以1000得m',negative_QC=True,positive_anomalies_retained=True))
qin=aligned(G,GM.loc['BOUNDARY_IN','source_name'],'瞬时流量').loc[formal].to_numpy();zout=aligned(G,GM.loc['BOUNDARY_OUT','source_name'],'闸前水位').loc[formal].to_numpy();assert np.isfinite(qin).all() and (qin>0).all() and np.isfinite(zout).all()
for tag,openings in [('C00_实际开度负值修复',actual),('C01_目标开度独立假设',target_open)]:
 raw=copy.deepcopy(base);m=raw if 'baseData'in raw else next(iter(raw.values()))
 for key,(typ,name) in m['timeSeries']['seriesLink'].items():
  if typ=='gates':v=openings[name]
  elif typ=='bleeders':v=bleed[name]
  elif typ=='nodes':v=qin if name=='NODE-main-0' else zout if name=='NODE-main-119' else None
  else:raise ValueError((typ,name))
  assert v is not None and len(v)==1441 and np.isfinite(v).all(),name
  m['timeSeries']['seriesData'][key]=[[int(t),float(x)] for t,x in zip(times,v)]
  initial_type='boundaries' if typ=='nodes' else typ
  if isinstance(m['initialData'].get(initial_type),dict):m['initialData'][initial_type][name]=float(v[0])
 m['timeControl'].update(startTime=0,calculateStartData=str(start),calculateStep=15,outputStep=1800,endTime=2592000)
 (R/'02_输入'/(tag+'_2025同工况30天.json')).write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
rows=[]
for station,row in GM.iterrows():
 if not station.startswith('JZZ'):continue
 g=G[G['站点名称']==row.source_name].set_index('时间').loc[formal]
 for col,metric in [('闸前水位','forebay_m'),('闸后水位','afterbay_m'),('瞬时流量','flow_m3s')]:
  for t,(dt,value) in zip(times,g[col].items()):rows.append(dict(station_code=station,sheet=row.source_name,timestamp=str(dt),time_s=int(t),metric=metric,value=float(value),valid=bool(np.isfinite(value) and (value>0 if metric!='flow_m3s' else value>=0))))
pd.DataFrame(rows).to_csv(R/'02_输入/实测评分长表.csv',index=False)
pd.DataFrame(fixes).to_csv(R/'03_运行参数/2025输入补值逐点记录.csv',index=False)
(R/'03_运行参数/2025开度归一化上限.json').write_text(json.dumps(limits,ensure_ascii=False,indent=2)+'\n')
(R/'03_运行参数/2025时序来源映射.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
summary=dict(start=str(start),end=str(end),compute_dt_s=15,output_dt_s=1800,series=313,points_per_series=1441,total_points=313*1441,gate_holes=len(actual),offtakes=len(bleed),source_offtake_names=len(used),upstream='陶岔渠首引水闸瞬时流量，不再使用刁河替代',downstream='北拒马河暗渠进口节制闸闸前水位',fixes=len(fixes),fixes_by_kind=pd.DataFrame(fixes).groupby('kind').size().to_dict(),net_inflow_mean=float(np.mean(qin-sum(bleed.values()))),net_inflow_min=float(np.min(qin-sum(bleed.values()))),raw_scoring=True,assumptions=['白庄长坏段按其所有有效观测0补值','袁庄原约2h有效观测线性对齐30min；吴庄短坏点同法','西黑山进口闸作为FSK121别名，文村北不重复扣','上街小幅负分水原样保留，待核是否逆流或零漂','部分2025未绑定源站位置不明确，不擅自加入主渠','正实际开度疑似异常保留，目标开度仅独立假设','归一化高度覆盖开度，不宣称物理最大行程已核准'])
(R/'03_运行参数/2025输入构造摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
