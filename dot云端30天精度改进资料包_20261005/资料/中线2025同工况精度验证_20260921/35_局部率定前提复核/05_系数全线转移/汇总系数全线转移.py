from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';M00=R/'34_剩余项目逐项对照/02_输入敏感性/M00_原输入完整30天';names=['X00_原系数含测点30天','X06_仅转移十二里河系数30天','X16_仅转移兰河系数30天','X616_同时转移两闸系数30天'];obsfile=B/'共享输入/作者已核实修正观测层.csv';obs=pd.read_csv(obsfile);obs=obs[obs.valid&obs.metric.isin(['forebay_m','afterbay_m'])&(obs.time_s>0)].rename(columns={'station_code':'station','value':'observed'});cfgs={};runs={};checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in names:
 d=O/name;status=json.loads((d/'进度.json').read_text());cfg=json.loads((d/'运行配置.json').read_text());cfgs[name]=cfg;assert status['status']=='COMPLETED' and status['completed_cycles']==1440 and not(d/'失败.json').exists();raw=pd.read_csv(d/'半小时水位流量.csv');assert len(raw)==59*1440 and raw.time_s.nunique()==1440 and raw.time_s.min()==1800 and raw.time_s.max()==2592000 and not raw[['station','time_s']].duplicated().any();mapped=pd.read_csv(d/'实际位置水位.csv');assert len(mapped)==43*1440 and mapped.valid.all() and mapped.time_s.nunique()==1440 and not mapped[['station','metric','time_s']].duplicated().any();assert mapped.groupby('group').apply(lambda z:z[['station','metric']].drop_duplicates().shape[0],include_groups=False).to_dict()=={'旧参与同化测点':34,'旧留出检验测点':9}
 nodes=raw.melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['station','metric','time_s','observed','pdf_comparable']],on=['station','metric','time_s'],validate='one_to_one');assert len(nodes)==118*1440;nodes['error']=nodes.sim-nodes.observed;mapped['error']=mapped.mapped_sim-mapped.observed;runs[name]=(nodes,mapped)
 with np.load(d/'末态.npz')as z:assert len(z.files)==6 and all(np.isfinite(z[k]).all()for k in z.files)
 assert sha(B/'共享输入/C07.json')==cfg['input_sha256'] and sha(B/'共享输入/初始化后状态_0s.npz')==cfg['initial_sha256'];checks.append(dict(name=name,status=status,configuration=cfg,rows_nodes=len(raw),rows_mapped=len(mapped),final_six_states_finite=True))
control=pd.read_csv(O/names[0]/'半小时水位流量.csv');old=pd.read_csv(M00/'半小时水位流量.csv');assert control.equals(old)
with np.load(O/names[0]/'末态.npz')as z,np.load(M00/'末态.npz')as oldz:final_equal={key:bool(np.array_equal(z[key],oldz[key]))for key in z.files};assert all(final_equal.values())
assert all(cfgs[name]['initial_state_sha256']==cfgs[names[0]]['initial_state_sha256']for name in names)
periods=[('前15天',0,1296000),('后15天',1296000,2592000),('完整30天',0,2592000)];tot=[];pts=[];impact=[];nei=[];stations=['JZZ5','JZZ6','JZZ7','JZZ15','JZZ16','JZZ17']
for name,(nodes,mapped)in runs.items():
 for period,lo,hi in periods:
  nd=nodes[nodes.time_s.between(lo,hi,inclusive='right')];md=mapped[mapped.time_s.between(lo,hi,inclusive='right')]
  for scope,g in [('闸节点_全线118测点',nd),('闸节点_作者可比88测点',nd[nd.pdf_comparable]),('实际位置_旧参与同化34测点',md[md.group=='旧参与同化测点']),('实际位置_旧留出9测点',md[md.group=='旧留出检验测点']),('实际位置_全部43测点',md)]:
   tot.append(dict(run=name,period=period,scope=scope,n=len(g),MAE=float(g.error.abs().mean()),RMSE=float(np.sqrt((g.error**2).mean())),bias=float(g.error.mean()),max_abs=float(g.error.abs().max())))
  for location_type,data in [('闸节点',nd),('实际位置',md)]:
   for (station,metric),g in data.groupby(['station','metric']):pts.append(dict(run=name,period=period,location_type=location_type,station=station,metric=metric,n=len(g),MAE=float(g.error.abs().mean()),RMSE=float(np.sqrt((g.error**2).mean())),bias=float(g.error.mean()),max_abs=float(g.error.abs().max())))
   baseline=runs[names[0]][0 if location_type=='闸节点'else 1];simcol='sim'if location_type=='闸节点'else'mapped_sim';b=baseline[baseline.time_s.between(lo,hi,inclusive='right')];pair=data.merge(b[['station','metric','time_s',simcol]],on=['station','metric','time_s'],suffixes=('','_base'),validate='one_to_one');pair['delta']=pair[simcol]-pair[simcol+'_base']
   for(station,metric),g in pair.groupby(['station','metric']):impact.append(dict(run=name,period=period,location_type=location_type,station=station,metric=metric,mean_water_change_m=float(g.delta.mean()),max_abs_water_change_m=float(g.delta.abs().max())))
S=pd.DataFrame(tot);P=pd.DataFrame(pts);I=pd.DataFrame(impact);S.to_csv(O/'四组分时段精度汇总.csv',index=False);P.to_csv(O/'四组逐测点精度.csv',index=False);I.to_csv(O/'四组水位变化.csv',index=False)
for name in names:
 for period,_,_ in periods:
  for station in stations:
   for metric in ['forebay_m','afterbay_m']:
    for loc in ['闸节点','实际位置']:
     g=P[(P.run==name)&(P.period==period)&(P.station==station)&(P.metric==metric)&(P.location_type==loc)]
     if len(g):nei.append({**g.iloc[0].to_dict(),'status':'可评分'})
     else:nei.append(dict(run=name,period=period,location_type=loc,station=station,metric=metric,status='无冻结43点对应算子，不填0也不以闸节点冒充'))
pd.DataFrame(nei).to_csv(O/'十二里河兰河及邻站精度.csv',index=False)
counts=[]
for period,_,_ in periods:
 for loc in ['闸节点','实际位置']:
  base=P[(P.run==names[0])&(P.period==period)&(P.location_type==loc)]
  for name in names[1:]:
   a=P[(P.run==name)&(P.period==period)&(P.location_type==loc)].merge(base[['station','metric','MAE']],on=['station','metric'],suffixes=('','_base'));diff=a.MAE-a.MAE_base;counts.append(dict(run=name,period=period,location_type=loc,total_points=len(a),improved_gt_1cm=int((diff<-.01).sum()),worsened_gt_1cm=int((diff>.01).sum()),within_1cm=int((diff.abs()<=.01).sum())))
pd.DataFrame(counts).to_csv(O/'改善与恶化测点数量.csv',index=False)
# 半小时测点值由同一冻结观测算子生成，末态独立重算终点43点以核查索引/偏移。
ops=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());mapped_final=[]
for name in names:
 with np.load(O/name/'末态.npz')as z:
  tail=runs[name][1].query('time_s==2592000').set_index(['station','metric']);maxdiff=0
  for op in ops:
   value=float(z[op['state']][op['indices']]@np.array(op['weights'])+op['offset']);maxdiff=max(maxdiff,abs(value-tail.loc[(op['station'],op['metric']),'mapped_sim']))
  assert maxdiff<1e-12;mapped_final.append(dict(run=name,max_end_value_reconstruction_error_m=maxdiff))
result=dict(all_four_complete=True,cycles=1440,control_nodes_equal_M00=True,control_final_state_equal_M00=final_equal,initial_states_equal=True,only_listed_coefficients_changed=True,all_coefficients_fixed_during_run=True,geometry_unchanged=True,observations_not_updated=True,baseline_not_replaced=True,checks=checks,mapped_final_reconstruction=mapped_final)
(O/'完整性与单因素转移核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(S[S.period=='后15天'].to_string(index=False));print(P[(P.period=='后15天')&(P.location_type=='实际位置')&(P.station.isin(['JZZ6','JZZ16']))].to_string(index=False));print(pd.DataFrame(counts).query('period=="后15天"').to_string(index=False))
# 有符号代数分解，不能把MAE相加，也不称测点间落差为已换算有效闸门水头。
decomp=[];decomp_stats=[]
for name,(_,mapped)in runs.items():
 for station in ['JZZ6','JZZ16']:
  front=mapped[(mapped.station==station)&(mapped.metric=='forebay_m')][['time_s','mapped_sim','observed']];back=mapped[(mapped.station==station)&(mapped.metric=='afterbay_m')][['time_s','mapped_sim','observed']];pair=front.merge(back,on='time_s',suffixes=('_front','_back'),validate='one_to_one');pair['forebay_error_m']=pair.mapped_sim_front-pair.observed_front;pair['afterbay_error_m']=pair.mapped_sim_back-pair.observed_back;pair['model_point_level_difference_m']=pair.mapped_sim_front-pair.mapped_sim_back;pair['observed_point_level_difference_m']=pair.observed_front-pair.observed_back;pair['point_difference_error_m']=pair.model_point_level_difference_m-pair.observed_point_level_difference_m;pair['identity_residual_m']=pair.forebay_error_m-pair.afterbay_error_m-pair.point_difference_error_m;assert pair.identity_residual_m.abs().max()<1e-12;pair['opposite_sign']=pair.afterbay_error_m*pair.point_difference_error_m<0;pair['cancelled_component_m']=np.where(pair.opposite_sign,np.minimum(pair.afterbay_error_m.abs(),pair.point_difference_error_m.abs()),0.);pair['run']=name;pair['station']=station;decomp.append(pair)
  for period,lo,hi in periods:
   g=pair[pair.time_s.between(lo,hi,inclusive='right')];decomp_stats.append(dict(run=name,station=station,period=period,n=len(g),forebay_bias_m=float(g.forebay_error_m.mean()),afterbay_bias_m=float(g.afterbay_error_m.mean()),point_level_difference_bias_m=float(g.point_difference_error_m.mean()),forebay_MAE_m=float(g.forebay_error_m.abs().mean()),afterbay_MAE_m=float(g.afterbay_error_m.abs().mean()),point_level_difference_MAE_m=float(g.point_difference_error_m.abs().mean()),opposite_sign_fraction=float(g.opposite_sign.mean()),mean_cancelled_component_m=float(g.cancelled_component_m.mean()),identity_max_abs_residual_m=float(g.identity_residual_m.abs().max())))
pd.concat(decomp,ignore_index=True).to_csv(O/'两闸实际位置逐时误差代数分解.csv',index=False);pd.DataFrame(decomp_stats).to_csv(O/'两闸实际位置分时段误差分解汇总.csv',index=False);print(pd.DataFrame(decomp_stats).query('period=="后15天"').to_string(index=False))
