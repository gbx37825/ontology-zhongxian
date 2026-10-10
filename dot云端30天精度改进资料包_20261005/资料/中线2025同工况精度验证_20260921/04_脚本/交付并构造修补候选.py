"""逐点变更可审计；仅新增候选，不覆盖原文件/冻结基线。"""
from pathlib import Path
import copy,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'14_输入修补交付';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=R/'02_输入/C00_实际开度负值修复_2025同工况30天.json';baseline_sha_before=sha(base);original=json.loads(base.read_text());m=next(iter(original.values()));events=pd.read_csv(R/'12_全量实测质量核查/开度突变含延迟响应核查.csv');raw=pd.read_csv(R/'12_全量实测质量核查/节制闸_全量原值.csv.gz',parse_dates=['时间']);start=pd.Timestamp('2025-07-22 12:00');end=pd.Timestamp('2025-08-21 12:00')
link={name:k for k,(typ,name) in m['timeSeries']['seriesLink'].items() if typ=='gates'};decisions=[];selected=[]
for ix,row in events.iterrows():
 short=row.event_type=='2小时内回归的短脉冲';strict=row.classification.startswith('目标平稳');model=str(row.code).startswith('JZZ');manual=row.code=='JZZ42' and row.peak_time in ['2025-07-22 16:30:00','2025-07-22 17:30:00']
 if not row.in_author_30d:decision='归档待后续时段处理；不在当前30天输入';version=''
 elif not model:decision='不改开度：当前模型未消费该站开度';version=''
 elif strict and short:decision='纳入C02/C03：短脉冲插值假设';version='C02+C03';selected.append((ix,row,False))
 elif manual:decision='仅纳入C03：人工重点脉冲插值假设';version='C03';selected.append((ix,row,True))
 else:decision='暂不删除：长异常、真实响应或证据不足';version=''
 decisions.append(dict(event_id=f'E{ix+1:04d}',**row.to_dict(),decision=decision,candidate=version))
pd.DataFrame(decisions).to_csv(O/'全量事件处置清单.csv',index=False)
changes=[]
for ix,ev,manual in selected:
 name=next(n for n in link if n.startswith(ev.code+'-') and n.endswith('-'+str(int(ev.hole))));g=raw[raw.站点名称==ev.station].set_index('时间').sort_index();col=f'孔{int(ev.hole)}实际开度';target=f'孔{int(ev.hole)}目标开度';left=pd.Timestamp(ev.start)-pd.Timedelta(minutes=30);right=pd.Timestamp(ev.end);idx=pd.date_range(ev.start,right-pd.Timedelta(minutes=30),freq='30min')
 assert len(idx)>0 and (right-left).total_seconds()<=7200
 av=float(g.loc[left,col])/1000;bv=float(g.loc[right,col])/1000;assert min(av,bv)>=0 and abs(av-bv)<=.1
 tv=g.loc[left:right,target].to_numpy(float)/1000;assert np.isfinite(tv).all() and np.ptp(tv)<=.1
 for t in idx:
  new=av+(bv-av)*(t-left).total_seconds()/(right-left).total_seconds();old=float(g.loc[t,col])/1000;second=int((t-start).total_seconds());pos=second//1800;key=link[name];assert m['timeSeries']['seriesData'][key][pos]==[second,old]
  changes.append(dict(event_id=f'E{ix+1:04d}',station=ev.station,station_code=ev.code,hole=int(ev.hole),name=name,series_key=key,timestamp=str(t),time_s=second,source_file='节制闸.xlsx',source_sheet=g.loc[t,'源工作表'],source_row=int(g.loc[t,'源行号']),raw_value_mm=old*1000,baseline_value_m=old,replacement_m=new,target_m=float(g.loc[t,target])/1000,left_time=str(left),left_value_m=av,right_time=str(right),right_value_m=bv,method='剔除候选点后，用同孔两侧有效端点按时间线性插值',certainty='高疑似，非现场故障确认',reason='人工重点：三孔目标0、开度大跳后回零；16:30存在其他孔操作及水位尖点，不作确定坏点' if manual else '短脉冲回到原基线，目标平稳，水位/流量邻域变化小',candidate='C03' if manual else 'C02+C03',manual_focus=manual))
pd.DataFrame(changes).to_csv(O/'新增修补逐点清单.csv',index=False)
# 已有负码处理也一并交付，明确长段补零仍是假设而非“已确认真值”。
inherited=pd.read_csv(R/'03_运行参数/2025输入补值逐点记录.csv');inherited['disposition']=np.where(inherited.reason.str.contains('全为0'),'继承旧候选的长坏段补零假设，仍待确认','继承C00的有效端点间插值');gm={x['station_code']:x['source_name'] for x in json.loads((R/'02_输入/2025节制闸站名映射.json').read_text())}
G=pd.read_csv(R/'02_输入/2025节制闸原始窗口.csv',parse_dates=['时间']).set_index(['站点名称','时间']);F=pd.read_csv(R/'02_输入/2025分退水原始窗口.csv',parse_dates=['时间']).set_index(['闸站名称','时间']);extra=[]
for x in inherited.itertuples():
 gate=x.kind=='实际开度';sn=gm[x.name.split('-')[0]] if gate else x.name;frame=G if gate else F;rr=frame.loc[(sn,pd.Timestamp(x.timestamp))];column=f'孔{x.name.rsplit("-",1)[1]}实际开度' if gate else '总瞬时流量'
 extra.append(dict(source_file='节制闸.xlsx' if gate else '分水口.xlsx',source_sheet='节制闸' if gate else '20250501-20250831',source_station=sn,source_row=int(rr['源行号']),source_column=column,unit='mm' if gate else 'm3/s'))
for k in extra[0]:inherited[k]=[v[k] for v in extra]
inherited.to_csv(O/'既有负码与缺测补值清单.csv',index=False)
validation=[]
for tag,include_manual in [('C02_高疑似短脉冲修补候选',False),('C03_含七里河重点脉冲修补候选',True)]:
 data=copy.deepcopy(original);mm=next(iter(data.values()));used=[d for d in changes if include_manual or not d['manual_focus']]
 for d in used:mm['timeSeries']['seriesData'][d['series_key']][d['time_s']//1800][1]=d['replacement_m']
 # 对所有数值逐点差分，确认只变更白名单，不动测量评分、边界、分水和初态。
 allowed={(d['series_key'],d['time_s']//1800) for d in used};diff=[]
 for k,arr in mm['timeSeries']['seriesData'].items():
  a=np.array(arr,float);b=np.array(m['timeSeries']['seriesData'][k],float);assert a.shape==(1441,2) and np.isfinite(a).all() and np.array_equal(a[:,0],np.arange(1441)*1800)
  for i in np.flatnonzero(a[:,1]!=b[:,1]):diff.append((k,int(i)));assert (k,int(i)) in allowed
 assert set(diff)==allowed
 ca=copy.deepcopy(mm);cb=copy.deepcopy(m);ca.pop('timeSeries');cb.pop('timeSeries');assert ca==cb
 path=O/(tag+'_2025同工况30天.json');path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 validation.append(dict(candidate=tag,path=str(path.resolve()),sha256=sha(path),changed_points=len(diff),changed_holes=len(set(d['name'] for d in used)),changed_events=len(set(d['event_id'] for d in used)),series=313,points_per_series=1441,finite=True,only_whitelisted_values_changed=True,initial_state_geometry_parameters_unchanged=True))
assert sha(base)==baseline_sha_before
manifest=dict(baseline=str(base.resolve()),baseline_sha256=sha(base),period=[str(start),str(end)],candidates=validation,events_decisions=pd.DataFrame(decisions).decision.value_counts().to_dict(),inherited_fixes=len(inherited),inherited_by_name=inherited.groupby('name').size().to_dict(),new_changes=changes,not_changed=['原始Excel','C00/C01','所有水位评分观测','糙率/闸流系数/初始化','归一化上限（仍用原值保持单因素）'],scope='候选处理已执行；尚非确认错误数据集或精度验收版本')
(O/'交付与输入核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['new_changes','inherited_by_name']},ensure_ascii=False,indent=2))
