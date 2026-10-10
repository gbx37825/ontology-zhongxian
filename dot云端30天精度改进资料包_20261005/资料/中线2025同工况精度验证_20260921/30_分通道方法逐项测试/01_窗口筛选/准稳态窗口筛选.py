"""预设阈值的只读筛选；不加载水力求解器，不变更输入、参数和基线。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];I=R/'26_冻结基线与后续归因/共享输入'
threshold=json.loads((O/'筛选阈值.json').read_text()); model=I/'C07.json';obsfile=I/'作者已核实修正观测层.csv'
M=json.loads(model.read_text())['中线全线'];B=M['baseData'];T=np.arange(1441)*1800
D={};typ={}
for sid,(kind,name) in M['timeSeries']['seriesLink'].items():
 a=np.asarray(M['timeSeries']['seriesData'][sid],float);assert np.array_equal(a[:,0],T)
 D[name]=a[:,1];typ[name]=kind
for kind in ['gates','bleeders']:
 for name,v in M['initialData'][kind].items():
  if name not in D:D[name]=np.full(len(T),float(v));typ[name]=kind
obs=pd.read_csv(obsfile);positions=pd.read_csv(R/'02_输入/测点位置映射.csv').set_index('stationcode')
Z={}
for (station,metric),g in obs.groupby(['station_code','metric']):
 g=g.set_index('time_s').reindex(T);Z[station+':'+metric]=g.value.where(g.valid.fillna(False)).to_numpy(float)
bleedx={name:float(b['channel'][1]) for name,b in B['bleeders'].items() if b.get('channel')}
gates=[n for n in D if typ[n]=='gates'];bleeds=[n for n in D if typ[n]=='bleeders'];nodeq=[n for n in D if typ[n]=='nodes' and B['nodes'][n]['nodeType']!=1];nodeh=[n for n in D if typ[n]=='nodes' and B['nodes'][n]['nodeType']==1]
assert len(nodeq)==1 and len(nodeh)==1
D['测流桩号未闭合']=np.full(len(T),np.nan)
records=[];details=[]
def score(names,source,l,r,kind):
 if not names:return (0.,'')
 a=np.array([source[n][l:r+1] for n in names]); ranges=np.ptp(a,axis=1)
 if kind=='flow':ranges/=np.maximum(np.abs(np.mean(a,axis=1)),threshold['流量相对极差分母下限m3s'])
 lim={'gate':.02,'flow':.02,'level':.03}[kind]; ratios=np.where(np.isfinite(ranges),ranges/lim,np.inf);j=int(np.argmax(ratios));return float(ratios[j]),names[j]
def add(scope,station,l,r,h,sets):
 rec={'scope':scope,'station':station,'window_h':h,'start_s':int(T[l]),'end_s':int(T[r]),'split':'训练' if T[r]<=1296000 else '保留' if T[l]>=1296000 else '跨界'}
 for key,names,src,kind in sets:
  value,worst=score(names,src,l,r,kind);rec[key+'_ratio']=value;rec[key+'_worst']=worst
 ratios=[v for k,v in rec.items() if k.endswith('_ratio')];rec['max_threshold_ratio']=max(ratios);rec['passed']=rec['max_threshold_ratio']<=1+1e-10;records.append(rec)
for h in threshold['窗口小时']:
 span=int(h*2)
 for r in range(span,len(T)):
  l=r-span
  add('全线输入','全线',l,r,h,[('gate',gates,D,'gate'),('inflow',nodeq,D,'flow'),('tailwater',nodeh,D,'level'),('offtake',bleeds,D,'flow')])
  for station in ['JZZ4','JZZ6','JZZ16']:
   n=int(station[3:]);lo=float(positions.loc[station,'meter_chainage']);hi=float(positions.loc[station,'gate_chainage']);localtakes=[x for x in bleeds if min(lo,hi)<bleedx.get(x,-1)<=max(lo,hi)]
   add('目标闸局部',station,l,r,h,[('gate',[k for k in gates if k.startswith(station+'-')],D,'gate'),('flow',[station+':flow_m3s'],Z,'flow'),('level',[station+':forebay_m',station+':afterbay_m'],Z,'level'),('offtake',localtakes,D,'flow')])
   adjacent=[f'JZZ{j}' for j in [n-1,n,n+1]];xa=float(positions.loc[adjacent[0],'meter_chainage']);xb=float(positions.loc[adjacent[-1],'meter_chainage']);takes=[x for x in bleeds if xa<bleedx.get(x,-1)<=xb] if np.isfinite(xa) and np.isfinite(xb) else ['测流桩号未闭合']
   add('扩展局部三闸',station,l,r,h,[('gate',[k for k in gates if any(k.startswith(c+'-') for c in adjacent)],D,'gate'),('flow',[c+':flow_m3s' for c in adjacent],Z,'flow'),('level',[c+':'+metric for c in adjacent for metric in ['forebay_m','afterbay_m']],Z,'level'),('offtake',takes,D,'flow')])
df=pd.DataFrame(records);df.to_csv(O/'全部窗口筛选.csv.gz',index=False)
summary=df[df.split!='跨界'].groupby(['scope','station','window_h','split']).agg(windows=('passed','size'),passed=('passed','sum'),best_threshold_ratio=('max_threshold_ratio','min')).reset_index();summary.to_csv(O/'窗口通过汇总.csv',index=False)
best=df[df.split!='跨界'].sort_values(['max_threshold_ratio','end_s']).groupby(['scope','station','window_h','split'],sort=False).head(1);best.to_csv(O/'各组最优候选.csv',index=False)
local=df.query("scope=='目标闸局部' and split=='训练' and passed").sort_values(['window_h','max_threshold_ratio','end_s'],ascending=[False,True,True])
selected=local.drop_duplicates('station').to_dict('records')
for row in selected:
 station=row['station'];h=row['window_h'];r=row['end_s'];same=df[(df.window_h==h)&(df.end_s==r)];row['global_at_same_window']=same[same.scope=='全线输入'].iloc[0].to_dict();row['extended_at_same_window']=same[(same.scope=='扩展局部三闸')&(same.station==station)].iloc[0].to_dict()
config={'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'observation_sha256':hashlib.sha256(obsfile.read_bytes()).hexdigest(),'input_counts':{k:sum(t==k for t in typ.values()) for k in set(typ.values())},'selection_policy':'训练期仅按固定波动门槛筛选，优先24h，其次门槛比值最小；不按模拟误差选窗','selected':selected,'summary':summary.to_dict('records')}
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
config.update({'field_calibration_eligibility': {'full_line_6h_or_24h_qualified': False, 'verified_extended_three_gate_qualified': False, 'any_target_gate_24h_qualified': False, 'direct_field_steady_calibration_ready': False, 'reason': '全线及扩展范围固定门槛未通过；JZZ16扩展上游测流桩号缺失。局部6小时波动小不证明现场稳态或流量闭合。'}, 'recommended_model_only_diagnostic': {'station': 'JZZ6', 'window_start_s': 153000, 'window_end_s': 174600, 'historical_24h_start_s': 88200, 'purpose': '模型内恒边界/提前24小时历史推进交叉核查与灵敏度，不称现场准稳态率定'}})
config=clean(config)
(O/'候选窗口摘要.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False))
print(summary.to_string(index=False));print('SELECTED',json.dumps(clean(selected),ensure_ascii=False,indent=2))
