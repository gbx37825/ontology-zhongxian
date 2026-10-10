"""不改原门槛；为(n4,C4,C5)联合灵敏度检查筛选时间工况，不率定。"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';T=np.arange(1441)*1800;old=R/'30_分通道方法逐项测试/01_窗口筛选/筛选阈值.json';th=json.loads(old.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
assert th['开度极差上限m']==.02 and th['流量相对极差上限']==.02 and th['水位极差上限m']==.03
policy=dict(purpose='只筛联合n4/C4/C5多工况灵敏度候选，不率定、不跑水力',upstream_Q='JZZ4测流48495.37m',downstream_H='JZZ5闸后水位74731m，边界不评分',targets=['JZZ4闸前48695.37m','JZZ4闸后48795.37m'],extra_Q_diagnostic='JZZ5测流74821m仅核查窗口稳定性，不作入口/出口边界，不当局部切口独立流量真值',internal_gates=['JZZ4所有孔','JZZ5所有孔'],threshold=th,threshold_sha256=sha(old),windows_h=[6,24],training_end_s=1296000,ranking='先通过原门槛，优先24h，再最大门槛比最小、结束时刻最早；不看任何仿真误差',nonoverlap='闭区间样本不得共用，第二窗start>第一窗end；按终点排序贪心给最多数量的非重叠时间窗，不代表不同工况')
dump(O/'联合诊断选窗规则.json',policy)
s=json.loads((B/'共享输入/C07.json').read_text());s=s if 'baseData'in s else next(iter(s.values()));D={};typ={}
for k,(kind,name) in s['timeSeries']['seriesLink'].items():
 a=np.asarray(s['timeSeries']['seriesData'][k]);assert np.array_equal(a[:,0],T);D[name]=a[:,1].astype(float);typ[name]=kind
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');Z={};V={}
for (st,metric),g in obs.groupby(['station_code','metric']):
 g=g.set_index('time_s').reindex(T);Z[(st,metric)]=g.value.to_numpy();V[(st,metric)]=g.valid.fillna(False).to_numpy(bool)
gates=[n for n in D if typ[n]=='gates' and (n.startswith('JZZ4-')or n.startswith('JZZ5-'))];takes=[n for n,b in s['baseData']['bleeders'].items()if b.get('channel')and 48495.37<float(b['channel'][1])<=74731];fields=[('JZZ4','flow_m3s'),('JZZ5','flow_m3s'),('JZZ4','forebay_m'),('JZZ4','afterbay_m'),('JZZ5','afterbay_m')];series=[(n,D[n],np.isfinite(D[n]),'gate')for n in gates]+[(n,D[n],np.isfinite(D[n]),'flow')for n in takes]+[(st+':'+mt,Z[(st,mt)],V[(st,mt)],'flow'if mt=='flow_m3s'else'level')for st,mt in fields];origin=pd.Timestamp(s['timeControl']['calculateStartData']);frame=pd.DataFrame(dict(time_s=T,timestamp=[str(origin+pd.Timedelta(seconds=int(t)))for t in T]))
for st,mt in fields:frame[st+'_'+mt]=Z[(st,mt)];frame[st+'_'+mt+'_valid']=V[(st,mt)]
for n in gates+takes:frame[n]=D[n]
frame[frame.time_s<=1296000].to_csv(O/'联合诊断前15天全部输入.csv',index=False)
rows=[]
for hours in [6,24]:
 span=hours*2
 for i in range(721-span):
  row=dict(window_h=hours,start_s=int(T[i]),end_s=int(T[i+span]),start_time=str(origin+pd.Timedelta(seconds=int(T[i]))),end_time=str(origin+pd.Timedelta(seconds=int(T[i+span]))));ratios=[]
  for label,v,valid,kind in series:
   z=v[i:i+span+1];den=max(abs(z.mean()),1.)if kind=='flow' else 1.;ratio=np.ptp(z)/den/{'gate':.02,'flow':.02,'level':.03}[kind] if valid[i:i+span+1].all()else float('inf');row[label+'_ratio']=float(ratio);ratios.append((ratio,label))
  worst=max(ratios);row.update(passed=bool(worst[0]<=1+1e-10),max_threshold_ratio=float(worst[0]),worst=worst[1]);rows.append(row)
a=pd.DataFrame(rows);a.to_csv(O/'联合诊断6小时24小时全部窗口.csv',index=False);passed=a[a.passed].sort_values(['window_h','max_threshold_ratio','end_s'],ascending=[False,True,True]);passed.to_csv(O/'联合诊断通过窗口.csv',index=False);nonoverlap=[];summary=[];best=[]
for h,g in a.groupby('window_h'):
 p=g[g.passed];end=-1
 for r in p.sort_values('end_s').to_dict('records'):
  if r['start_s']>end:nonoverlap.append(r);end=r['end_s']
 b=g.sort_values(['max_threshold_ratio','end_s']).iloc[0].to_dict();best.append(b);summary.append(dict(window_h=int(h),checked=len(g),passed=len(p),nonoverlap_count=sum(x['window_h']==h for x in nonoverlap),best_window=b))
pd.DataFrame(nonoverlap,columns=a.columns).to_csv(O/'联合诊断不重叠通过窗口.csv',index=False);pd.DataFrame(best).to_csv(O/'联合诊断最接近门槛窗口.csv',index=False);detail=[]
for r in best:
 mask=frame.time_s.between(r['start_s'],r['end_s']);frame[mask].to_csv(O/f'联合诊断最接近{r["window_h"]}小时窗输入.csv',index=False)
 for label,v,valid,kind in series:
  z=v[(T>=r['start_s'])&(T<=r['end_s'])];ratio=r[label+'_ratio'];detail.append(dict(window_h=r['window_h'],start_s=r['start_s'],end_s=r['end_s'],series=label,kind=kind,mean=float(z.mean()),range=float(np.ptp(z)),relative_range=float(np.ptp(z)/max(abs(z.mean()),1.))if kind=='flow'else None,threshold_ratio=ratio,passed=ratio<=1+1e-10))
pd.DataFrame(detail).to_csv(O/'联合诊断最接近窗口逐项失败原因.csv',index=False);source=json.loads((R/'03_运行参数/2025时序来源映射.json').read_text());result=dict(no_simulation=True,no_parameters_identified=True,original_threshold_unchanged=sha(old)==policy['threshold_sha256'],all_offtakes=[dict(name=n,location=s['baseData']['bleeders'][n]['channel'],source=next(x for x in source if x['target']==n))for n in takes],gates=gates,summary=summary,selected=None if len(passed)==0 else passed.iloc[0].to_dict(),input_sha256=sha(B/'共享输入/C07.json'),observation_sha256=sha(B/'共享输入/作者已核实修正观测层.csv'))
dump(O/'联合诊断选窗结果.json',result);print(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.DataFrame(detail).query('passed==False').to_string(index=False));print('offtakes',takes)
# 两工况完整输入及均值都保留，窗口排序不使用模型误差。
means=[]
for r in passed.itertuples():
 g=frame[frame.time_s.between(r.start_s,r.end_s)];g.to_csv(O/f'联合诊断通过窗_{r.start_s}_{r.end_s}秒输入.csv',index=False);record=dict(start_s=r.start_s,end_s=r.end_s)
 for col in g:
  if col in ['time_s','timestamp']or col.endswith('_valid'):continue
  record[col+'_arithmetic_mean']=float(g[col].mean());record[col+'_timeweighted_mean']=float(np.trapezoid(g[col],g.time_s)/(r.end_s-r.start_s))
 record['Q4_minus_Q5_minus_known_offtakes_without_storage_m3s']=record['JZZ4_flow_m3s_timeweighted_mean']-record['JZZ5_flow_m3s_timeweighted_mean']-record['FSK6-2-pool3_timeweighted_mean']-record['FSK7-1-pool4_timeweighted_mean'];means.append(record)
pd.DataFrame(means).to_csv(O/'联合诊断两工况均值.csv',index=False)
