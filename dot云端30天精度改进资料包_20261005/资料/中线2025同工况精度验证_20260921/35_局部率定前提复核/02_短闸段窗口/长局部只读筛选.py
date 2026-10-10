"""Q16测流断面至H17闸前的局部筛查；保留原数值门槛与选择规则。"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];F=R/'26_冻结基线与后续归因';old=R/'30_分通道方法逐项测试/01_窗口筛选';th=json.loads((old/'筛选阈值.json').read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert th['开度极差上限m']==.02 and th['流量相对极差上限']==.02 and th['水位极差上限m']==.03
policy=dict(upper_cut_m=300091.,lower_cut_m=327100.,boundary_Q='JZZ16 flow_m3s，上游安装测流断面',boundary_H='JZZ17 forebay_m，按安装桩号截断，边界水位禁止再评分',internal_water_targets=['JZZ16 forebay_m','JZZ16 afterbay_m'],window_hours=6,training_end_s=1296000,selection='同原T02：固定6h，先通过同一门槛，再按最大门槛比最小、结束时刻最早选择；不读取任何模拟误差',threshold_path=str(old/'筛选阈值.json'),threshold_sha256=sha(old/'筛选阈值.json'),original_global_result_unchanged=True,reason='重新定义有实测切口边界的局部子模型，不再要求域外JZZ15流量/开度和域外JZZ17闸门开度平稳；域内闸16及全部分水仍逐项检查',note='新局部适配，不宣称原PDF原样复现；通过波动筛查仍须局部双初值、提前起算、水量和几何范围校验')
qualification={'temporal_window_check': '通过原2%/0.02m/0.03m/6h门槛，仅时间波动通过', 'downstream_position_structure': '未核实：补充表绝对327100m与闸前约0.5m描述不一致；C07/建模表闸327420m、倒虹吸进口327407m', 'priority_table': '用户优先的位置统计表C19:E19只给退水闸→节制闸→流量计顺序，无水位绝对坐标，不能用它消除该矛盾', 'allowed_current_use': '按327100m假设做局部数值前提/敏感性诊断；不得宣称尾水边界现场位置已核实；不自动挪动水位测点', 'historical_sensor_identity': '2025水位历史列对应具体仪器未知'}
policy['qualification']=qualification
policy['boundary_H']='JZZ17 forebay_m按补充表候选绝对桩号327100m截断；位置文字/结构矛盾未闭合；边界水位禁止再评分'
dump(O/'长局部范围与筛选规则.json',policy)
a=json.loads((F/'共享输入/C07.json').read_text());s=a if 'baseData'in a else next(iter(a.values()));T=np.arange(1441)*1800;origin=pd.Timestamp(s['timeControl']['calculateStartData']);D={};typ={}
for k,(kind,name) in s['timeSeries']['seriesLink'].items():
 a=np.asarray(s['timeSeries']['seriesData'][k]);assert np.array_equal(a[:,0],T);D[name]=a[:,1].astype(float);typ[name]=kind
obs=pd.read_csv(F/'共享输入/作者已核实修正观测层.csv');Z={};V={}
for (st,metric),g in obs.groupby(['station_code','metric']):
 g=g.set_index('time_s').reindex(T);Z[(st,metric)]=g.value.to_numpy();V[(st,metric)]=g.valid.fillna(False).to_numpy(bool)
gates=[x for x in D if typ[x]=='gates' and x.startswith('JZZ16-')];takes=[x for x,v in s['baseData']['bleeders'].items() if v.get('channel') and 300091<float(v['channel'][1])<=327100];assert set(takes)=={'FSK28-2-pool15','FSK29-1-pool16','FSK30-2-pool16','FSK31-3-pool16'}
fields=[('JZZ16','flow_m3s'),('JZZ17','forebay_m'),('JZZ16','forebay_m'),('JZZ16','afterbay_m')];series=[(name,D[name],np.isfinite(D[name]),'gate')for name in gates]+[(name,D[name],np.isfinite(D[name]),'flow')for name in takes]+[(st+':'+metric,Z[(st,metric)],V[(st,metric)],'flow' if metric=='flow_m3s' else 'level') for st,metric in fields]
frame=pd.DataFrame(dict(time_s=T,timestamp=[str(origin+pd.Timedelta(seconds=int(t)))for t in T]))
for st,metric in fields:frame[st+'_'+metric]=Z[(st,metric)];frame[st+'_'+metric+'_valid']=V[(st,metric)]
for name in gates+takes:frame[name]=D[name]
frame.to_csv(O/'长局部_边界目标分水全时序.csv',index=False)
rows=[];details=[]
for i in range(len(T)-12):
 row=dict(start_s=int(T[i]),end_s=int(T[i+12]),start_time=str(origin+pd.Timedelta(seconds=int(T[i]))),end_time=str(origin+pd.Timedelta(seconds=int(T[i+12]))),split='训练' if T[i+12]<=1296000 else '保留' if T[i]>=1296000 else '跨界');rr=[]
 for label,v,valid,kind in series:
  z=v[i:i+13];den=max(abs(z.mean()),1.) if kind=='flow' else 1.;lim={'gate':.02,'flow':.02,'level':.03}[kind];ratio=float(np.ptp(z)/den/lim) if valid[i:i+13].all() else float('inf');row[label+'_ratio']=ratio;rr.append((ratio,label))
 worst=max(rr);row.update(max_threshold_ratio=worst[0],worst=worst[1],passed=bool(worst[0]<=1+1e-10));rows.append(row)
allrows=pd.DataFrame(rows);allrows.to_csv(O/'长局部_所有窗口筛选.csv',index=False);train=allrows[(allrows.split=='训练')&allrows.passed].sort_values(['max_threshold_ratio','end_s']);assert len(train)==4;train.to_csv(O/'长局部_四个通过训练窗口.csv',index=False);best=train.iloc[0].to_dict();assert best['start_s']==648000 and best['end_s']==669600
for r in train.itertuples():
 mask=frame.time_s.between(r.start_s,r.end_s);frame[mask].to_csv(O/f'长局部_通过窗_{r.start_s}_{r.end_s}秒.csv',index=False)
 for label,v,valid,kind in series:
  z=v[(T>=r.start_s)&(T<=r.end_s)];details.append(dict(start_s=r.start_s,end_s=r.end_s,series=label,kind=kind,min=float(z.min()),max=float(z.max()),mean=float(z.mean()),range=float(np.ptp(z)),relative_range=float(np.ptp(z)/max(abs(z.mean()),1.)) if kind=='flow' else None))
pd.DataFrame(details).to_csv(O/'长局部_四窗逐输入波动.csv',index=False)
ops=json.loads((F/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());sources=json.loads((R/'03_运行参数/2025时序来源映射.json').read_text());pos=pd.read_csv(R/'02_输入/优先级确认后测流映射.csv');qextra=pos[pos.precise_position_usable&(pos.usable_meter_chainage>300091)&(pos.usable_meter_chainage<327100)]
dump(O/'长局部_边界与全部分水映射.json',dict(qualification=qualification,policy=policy,selected=best,heldout_passed=int(((allrows.split=='保留')&allrows.passed).sum()),train_passed=4,gate_holes=gates,water_operators=[o for o in ops if (o['station'],o['metric'])in [('JZZ16','forebay_m'),('JZZ16','afterbay_m'),('JZZ17','forebay_m')]],offtakes=[dict(name=n,channel_and_chainage=s['baseData']['bleeders'][n]['channel'],source=next(x for x in sources if x['target']==n))for n in takes],additional_independent_internal_flow_stations=qextra.stationcode.tolist(),external_flow_note='JZZ17测流位于327550m，在下游边界327100m以外且跨过JZZ17闸，不能直接作为本局部出口流量真值。Q16为边界不能当独立验证流量。',hashes={'C07':sha(F/'共享输入/C07.json'),'observations':sha(F/'共享输入/作者已核实修正观测层.csv'),'threshold_after':sha(old/'筛选阈值.json')}))
print(train[['start_s','end_s','start_time','end_time','max_threshold_ratio','worst']].to_string(index=False));print('selected',best['start_s'],best['end_s'],'otherinternalflow',qextra.stationcode.tolist(),'takes',takes)
