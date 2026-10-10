"""沿用T02门槛，仅明确局部切口；不加载求解器，不修改输入与旧筛查。"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];F=R/'26_冻结基线与后续归因';old=R/'30_分通道方法逐项测试/01_窗口筛选'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
th=json.loads((old/'筛选阈值.json').read_text());assert th['开度极差上限m']==.02 and th['流量相对极差上限']==.02 and th['水位极差上限m']==.03
# 先固定选择方式和用途，随后只用输入波动筛选，不读任何模型误差。
policy=dict(threshold_source=str(old/'筛选阈值.json'),threshold_sha256=sha(old/'筛选阈值.json'),thresholds_unchanged=th,cut_selection='仅目标闸短段：上游测流安装断面到闸后水位安装断面；不要求切口外远端闸门稳定',window_hours=6,training_end_s=1296000,holdout_start_s=1296000,selection='沿用原T02：已知三站24h均无合格；6h训练窗口先过门槛，再取各项门槛归一化最大值最小，平手按结束时刻最早；不看模拟误差',fit_target='闸前水位；闸后水位是下游边界，不评分也不作第二份独立同化数据',fit_parameter='只识别本闸各孔共享C倍率；n及其他参数固定',output_status='只通过低波动筛选，不冒称现场稳态或正式率定通过')
dump(O/'筛查预设与选择规则.json',policy)
raw=json.loads((F/'共享输入/C07.json').read_text());s=raw if 'baseData'in raw else next(iter(raw.values()));T=np.arange(1441)*1800;origin=pd.Timestamp(s['timeControl']['calculateStartData']);D={};typ={}
for k,(kind,name) in s['timeSeries']['seriesLink'].items():
 a=np.asarray(s['timeSeries']['seriesData'][k],float);assert np.array_equal(a[:,0],T);D[name]=a[:,1];typ[name]=kind
obs=pd.read_csv(F/'共享输入/作者已核实修正观测层.csv');Z={};V={}
for (st,metric),g in obs.groupby(['station_code','metric']):
 g=g.set_index('time_s').reindex(T);Z[(st,metric)]=g.value.to_numpy();V[(st,metric)]=g.valid.fillna(False).to_numpy(bool)
ops=json.loads((F/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());pos=pd.read_csv(R/'02_输入/优先级确认后测流映射.csv').set_index('stationcode');series_sources=json.loads((R/'03_运行参数/2025时序来源映射.json').read_text());rules=[];allwindows=[];choice=[];components=[]
for station in ['JZZ4','JZZ6','JZZ16']:
 front=next(x for x in ops if x['station']==station and x['metric']=='forebay_m');tail=next(x for x in ops if x['station']==station and x['metric']=='afterbay_m');xm=float(pos.loc[station,'usable_meter_chainage']);xt=tail['chainage'];assert xm<front['chainage']<xt
 gates=[name for name in D if typ[name]=='gates' and name.startswith(station+'-')];takes=[name for name,b in s['baseData']['bleeders'].items() if b.get('channel') and xm<float(b['channel'][1])<=xt];g=s['baseData']['gates'][gates[0]];nodes=[g['node1'],g['node2']];gate_ends=[]
 for node in nodes:
  ends=[]
  for name,c in s['baseData']['channels'].items():
   if c['node1']==node:ends.append(dict(channel=name,section=c['sec_names'][0],chainage=s['baseData']['sections'][c['sec_names'][0]]['location']))
   if c['node2']==node:ends.append(dict(channel=name,section=c['sec_names'][-1],chainage=s['baseData']['sections'][c['sec_names'][-1]]['location']))
  gate_ends.append(dict(node=node,channel_ends=ends))
 sources={name:next((x for x in series_sources if x['target']==name),None) for name in takes}
 rule=dict(station=station,name=pos.loc[station,'name'],upstream_Q_chainage_m=xm,forebay_target_chainage_m=front['chainage'],downstream_H_chainage_m=xt,boundary_H_not_scored=True,front_operator=front,tail_operator=tail,model_gate_endpoints=gate_ends,internal_gate_holes=gates,interval_offtakes=[dict(name=n,location=s['baseData']['bleeders'][n]['channel'],source=sources[n]) for n in takes],assumptions='历史仪器身份及测点基准仍待确认；在当前已冻结映射下作条件性候选，不用模型估计流量冒充独立实测')
 rules.append(rule);frame=pd.DataFrame(dict(time_s=T,timestamp=[str(origin+pd.Timedelta(seconds=int(t))) for t in T],upstream_Q_m3s=Z[(station,'flow_m3s')],downstream_boundary_H_m=Z[(station,'afterbay_m')],forebay_scoring_H_m=Z[(station,'forebay_m')],Q_valid=V[(station,'flow_m3s')],tail_H_valid=V[(station,'afterbay_m')],front_H_valid=V[(station,'forebay_m')]))
 for name in gates+takes:frame[name]=D[name]
 frame.to_csv(O/(station+'_局部边界与目标全时序.csv'),index=False)
 series=[(n,D[n],np.isfinite(D[n]),'gate') for n in gates]+[(n,D[n],np.isfinite(D[n]),'flow') for n in takes]+[(station+':flow_m3s',Z[(station,'flow_m3s')],V[(station,'flow_m3s')],'flow'),(station+':forebay_m',Z[(station,'forebay_m')],V[(station,'forebay_m')],'level'),(station+':afterbay_m',Z[(station,'afterbay_m')],V[(station,'afterbay_m')],'level')]
 rows=[]
 for i in range(len(T)-12):
  ratios=[]
  for label,v,valid,kind in series:
   z=v[i:i+13];rel=np.ptp(z)/max(abs(z.mean()),1.) if kind=='flow' else np.ptp(z);ratio=rel/({'gate':.02,'flow':.02,'level':.03}[kind]) if valid[i:i+13].all() else np.inf;ratios.append((ratio,label))
  worst=max(ratios);split='训练' if T[i+12]<=1296000 else '保留' if T[i]>=1296000 else '跨界';rows.append(dict(station=station,start_s=int(T[i]),end_s=int(T[i+12]),split=split,passed=bool(worst[0]<=1+1e-10),max_threshold_ratio=float(worst[0]),worst=worst[1]))
 a=pd.DataFrame(rows);allwindows+=rows;train=a[(a.split=='训练')&a.passed].sort_values(['max_threshold_ratio','end_s']);assert len(train)>0;chosen=train.iloc[0].to_dict();chosen.update(train_passed=len(train),holdout_passed=int(((a.split=='保留')&a.passed).sum()),first_other_nonoverlap_rule='与辨识窗包含的13个时刻无交叠，即end<selected_start或start>selected_end',not_used_for_fit_first15_count=int(((a.split=='训练')&a.passed&((a.end_s<chosen['start_s'])|(a.start_s>chosen['end_s']))).sum()));choice.append(chosen)
 keep=frame.time_s.between(chosen['start_s'],chosen['end_s']);frame[keep].to_csv(O/(station+'_固定辨识窗输入.csv'),index=False)
 for label,v,valid,kind in series:
  z=v[(T>=chosen['start_s'])&(T<=chosen['end_s'])];components.append(dict(station=station,series=label,kind=kind,min=float(z.min()),max=float(z.max()),mean=float(z.mean()),range=float(np.ptp(z)),relative_range=float(np.ptp(z)/max(abs(z.mean()),1.)) if kind=='flow' else None))
 oldrow=pd.read_csv(old/'各组最优候选.csv');o=oldrow[(oldrow.scope=='目标闸局部')&(oldrow.station==station)&(oldrow.window_h==6)&(oldrow.split=='训练')].iloc[0];assert chosen['start_s']==o.start_s and chosen['end_s']==o.end_s
pd.DataFrame(allwindows).to_csv(O/'三个短闸段全窗口筛选.csv',index=False);pd.DataFrame(choice).to_csv(O/'固定选择训练窗口.csv',index=False);pd.DataFrame(components).to_csv(O/'固定窗口逐输入波动.csv',index=False);dump(O/'局部边界与分水映射.json',rules);dump(O/'筛查完成核验.json',dict(read_only=True,solver_not_loaded=True,original_threshold_unchanged=sha(old/'筛选阈值.json')==policy['threshold_sha256'],chosen_windows_match_original_T02=True,inputs_sha256=sha(F/'共享输入/C07.json'),observations_sha256=sha(F/'共享输入/作者已核实修正观测层.csv'),selected=choice))
print(json.dumps(choice,ensure_ascii=False,indent=2));print(json.dumps(rules,ensure_ascii=False,indent=2))
