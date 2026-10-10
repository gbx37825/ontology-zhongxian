"""复用已存时序的沿程代数归因，不运行水力模型、不改观测。"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'35_局部率定前提复核/05_系数全线转移';B=R/'26_冻结基线与后续归因';names=['X00_原系数含测点30天','X16_仅转移兰河系数30天'];stations=[f'JZZ{i}'for i in range(16,26)];obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv').rename(columns={'station_code':'station','value':'observed'});obs=obs[obs.station.isin(stations)&obs.metric.isin(['forebay_m','afterbay_m'])&(obs.time_s>0)&obs.valid];station_names=pd.read_csv(R/'02_输入/2025节制闸站名映射.csv').set_index('station_code').model_name.to_dict();ops=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());opmap={(o['station'],o['metric']):o for o in ops if o['station']in stations};periods=[('前15天',0,1296000),('后15天',1296000,2592000),('完整30天',0,2592000)];datasets={};hashes={}
for run in names:
 raw=pd.read_csv(D/run/'半小时水位流量.csv');nodes=raw[raw.station.isin(stations)].melt(id_vars=['time_s','station'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['time_s','station','metric','observed','pdf_comparable']],on=['time_s','station','metric'],validate='one_to_one');mapped=pd.read_csv(D/run/'实际位置水位.csv');mapped=mapped[mapped.station.isin(stations)&mapped.valid].rename(columns={'mapped_sim':'sim'});assert len(nodes)==20*1440 and len(mapped)==7*1440
 for typ,a in [('闸节点',nodes),('安装位置候选',mapped)]:
  a=a.copy();a['error']=a.sim-a.observed;a['point']=a.station+'_'+a.metric;datasets[(run,typ)]=a
 for fn in ['半小时水位流量.csv','实际位置水位.csv']:
  f=D/run/fn;hashes[str(f.relative_to(R))]=hashlib.sha256(f.read_bytes()).hexdigest()
metadata=[]
for station in stations:
 for metric in ['forebay_m','afterbay_m']:
  op=opmap.get((station,metric));metadata.append(dict(station=station,name=station_names[station],metric=metric,position_operator_available=op is not None,chainage_m=None if op is None else op['chainage'],role='未进入冻结43点集合'if op is None else'原参与同化测点'if op['assimilated']else'原留出检验测点',position_flag='绝对327100m与闸前约0.5m文字、闸327420m/倒虹吸进口327407m冲突'if station=='JZZ17'and metric=='forebay_m'else'仍为冻结候选算子，历史仪器身份未全部确认'if op else'不填0、不以闸节点冒充安装位置'))
pd.DataFrame(metadata).to_csv(O/'测点覆盖与角色.csv',index=False)
pointstats=[];pairstats=[];hourly=[];influences=[];groupstats=[];checks=[]
nodepoints=[(s,m)for s in stations for m in ['forebay_m','afterbay_m']];mappedpoints=sorted(opmap,key=lambda x:opmap[x]['chainage'])
for(run,typ),a in datasets.items():
 points=nodepoints if typ=='闸节点'else mappedpoints;edge_specs=[]
 if typ=='闸节点':
  for i in range(16,26):edge_specs.append((f'JZZ{i}','forebay_m',f'JZZ{i}','afterbay_m','同闸前后',f'JZZ{i}闸门记录对；含测点与闸节点位置口径影响'))
  for i in range(16,25):edge_specs.append((f'JZZ{i}','afterbay_m',f'JZZ{i+1}','forebay_m','渠池两端记录',f'第{i}渠池名义两端，含区间渠道和建筑物；不是单一糙率或损失观测'))
 else:
  for (us,um),(ds,dm)in zip(points[:-1],points[1:]):
   typedge='同闸两测点'if us==ds else'跨缺点混合区间';note='只为两测点间水位差，未转换为闸门有效水头'if us==ds else'跨越缺失安装算子的闸后或中间闸站，不能定位到唯一渠池/设施'
   if us=='JZZ16'and um=='afterbay_m'and ds=='JZZ17':typedge='位置冲突区间';note='JZZ17尾点位置与结构冲突；不当作已核实第16渠池完整两端'
   if us=='JZZ17':note+='；上游JZZ17位置冲突'
   edge_specs.append((us,um,ds,dm,typedge,note))
 edge_error=[]
 for us,um,ds,dm,edge_type,note in edge_specs:
  u=a[(a.station==us)&(a.metric==um)][['time_s','sim','observed','error']];v=a[(a.station==ds)&(a.metric==dm)][['time_s','sim','observed','error']];pair=u.merge(v,on='time_s',suffixes=('_up','_down'),validate='one_to_one');assert len(pair)==1440;pair['model_level_difference_m']=pair.sim_up-pair.sim_down;pair['observed_level_difference_m']=pair.observed_up-pair.observed_down;pair['level_difference_error_m']=pair.model_level_difference_m-pair.observed_level_difference_m;pair['error_change_downstream_m']=pair.error_down-pair.error_up;pair['identity_residual_m']=pair.error_up-pair.error_down-pair.level_difference_error_m;assert pair.identity_residual_m.abs().max()<1e-12;pair['run']=run;pair['口径']=typ;pair['upstream']=us+'_'+um;pair['downstream']=ds+'_'+dm;pair['edge_type']=edge_type;hourly.append(pair);edge_error.append(pair[['time_s','level_difference_error_m']].set_index('time_s').level_difference_error_m)
  for period,lo,hi in periods:
   g=pair[pair.time_s.between(lo,hi,inclusive='right')];pairstats.append(dict(run=run,口径=typ,period=period,upstream=us+'_'+um,downstream=ds+'_'+dm,edge_type=edge_type,n=len(g),up_bias_m=float(g.error_up.mean()),down_bias_m=float(g.error_down.mean()),downstream_bias_increment_m=float(g.error_change_downstream_m.mean()),point_level_difference_bias_m=float(g.level_difference_error_m.mean()),point_level_difference_MAE_m=float(g.level_difference_error_m.abs().mean()),point_level_difference_max_abs_m=float(g.level_difference_error_m.abs().max()),identity_max_residual_m=float(g.identity_residual_m.abs().max()),note=note))
 start=a[(a.station==points[0][0])&(a.metric==points[0][1])].set_index('time_s').error;end=a[(a.station==points[-1][0])&(a.metric==points[-1][1])].set_index('time_s').error;tel=start-end-sum(edge_error);assert tel.abs().max()<1e-12;checks.append(dict(run=run,口径=typ,points=len(points),edges=len(edge_specs),all_edges_cycles=1440,telescoping_max_abs_residual_m=float(tel.abs().max()),end_point=points[-1]))
 reference=datasets[(names[0],typ)];pairs=a.merge(reference[['station','metric','time_s','sim']],on=['station','metric','time_s'],suffixes=('','_base'),validate='one_to_one');pairs['change']=pairs.sim-pairs.sim_base
 for period,lo,hi in periods:
  q=a[a.time_s.between(lo,hi,inclusive='right')];groups=[('本区20个闸节点记录',q)]if typ=='闸节点'else[('本区7个安装位置候选，全属原34参与组',q)]
  if typ=='闸节点':groups.append(('本区作者可比闸节点子集',q[q.pdf_comparable]))
  for scope,g in groups:groupstats.append(dict(run=run,口径=typ,period=period,scope=scope,points=g[['station','metric']].drop_duplicates().shape[0],samples=len(g),MAE=float(g.error.abs().mean()),bias=float(g.error.mean()),RMSE=float(np.sqrt((g.error**2).mean())),max_abs=float(g.error.abs().max())))
  for(station,metric),g in q.groupby(['station','metric']):pointstats.append(dict(run=run,口径=typ,period=period,station=station,name=station_names[station],metric=metric,n=len(g),bias_m=float(g.error.mean()),MAE_m=float(g.error.abs().mean()),max_abs_m=float(g.error.abs().max())))
  for(station,metric),g in pairs[pairs.time_s.between(lo,hi,inclusive='right')].groupby(['station','metric']):influences.append(dict(run=run,口径=typ,period=period,station=station,metric=metric,mean_water_change_m=float(g.change.mean()),max_abs_water_change_m=float(g.change.abs().max())))
P=pd.DataFrame(pointstats);S=pd.DataFrame(pairstats);I=pd.DataFrame(influences);G=pd.DataFrame(groupstats);P.to_csv(O/'沿程逐点偏差与MAE.csv',index=False);S.to_csv(O/'同闸与渠池端点水位差误差.csv',index=False);I.to_csv(O/'改变C16对沿程水位影响.csv',index=False);G.to_csv(O/'本区各口径分组统计.csv',index=False);pd.concat(hourly,ignore_index=True).to_csv(O/'逐时水位差与误差恒等式.csv.gz',index=False,compression='gzip')
base=P[P.run==names[0]];candidate=P[P.run==names[1]];compare=candidate.merge(base[['口径','period','station','metric','bias_m','MAE_m']],on=['口径','period','station','metric'],suffixes=('_C16','_base'));compare['MAE_change_m']=compare.MAE_m_C16-compare.MAE_m_base;compare['bias_change_m']=compare.bias_m_C16-compare.bias_m_base;compare['remaining_abs_bias_ratio']=compare.bias_m_C16.abs()/compare.bias_m_base.abs();compare.to_csv(O/'原C与改C16逐点比较.csv',index=False)
(O/'逐时恒等式与数据核验.json').write_text(json.dumps(dict(source_hashes=hashes,checks=checks,source_operators=7,nominal_gate_measurements=20,all7_original_assimilation_points=True,none_of_original_9_holdout_in_region=True,observations_not_changed=True,simulation_not_run=True),ensure_ascii=False,indent=2));print(S[(S.run==names[0])&(S.period=='后15天')].to_string(index=False));print(compare[(compare.period=='后15天')&(compare.口径=='安装位置候选')].to_string(index=False));print(G[G.period=='后15天'].to_string(index=False))
