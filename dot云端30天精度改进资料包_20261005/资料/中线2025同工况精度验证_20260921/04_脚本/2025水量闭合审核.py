"""只读水量闭合；不率定、不改节点，不以水位评分反调参数。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1];I=R/'02_输入';O=R/'07_对比分析';P=R/'03_运行参数'
path=I/'C00_实际开度负值修复_2025同工况30天.json';M=json.loads(path.read_text())['中线全线'];B=M['baseData'];start=pd.Timestamp('2025-07-22 12:00');end=pd.Timestamp('2025-08-21 12:00');split=start+pd.Timedelta(hours=72);idx=pd.date_range(start,end,freq='30min');F=pd.read_csv(I/'2025节制闸原始窗口.csv',parse_dates=['时间']);mapping=pd.read_csv(I/'2025节制闸站名映射.csv').set_index('station_code');pos=pd.read_csv(I/'优先级确认后测流映射.csv');gatepos=pd.read_csv(I/'测点位置映射.csv').set_index('stationcode');anchor=pd.read_csv(R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/2019水量闭合水位锚点.csv').set_index(['station','side'])
def obs(code,col):return F[F.站点名称==mapping.loc[code,'source_name']].set_index('时间').loc[idx,col].to_numpy(float)
qin=obs('BOUNDARY_IN','瞬时流量');qout=obs('BOUNDARY_OUT','瞬时流量');bleed={n:np.array(M['timeSeries']['seriesData'][s],float)[:,1] for s,(t,n) in M['timeSeries']['seriesLink'].items() if t=='bleeders'}
bleedx={n:float(B['bleeders'][n]['channel'][1]) for n in bleed};xmax=max(s['location'] for s in B['sections'].values());qxout=1197369. # 附件791行末流量计，不与末水位1197636混同。
# 两端差额分开：全模型边界以近末端流量计代出流含267m位置差；43个相邻已知测流区间用实际测流桩号。
gatex=[0.]+[float(gatepos.loc[f'JZZ{k}','gate_chainage']) for k in range(2,61)]+[xmax];codes=['BOUNDARY_IN']+[f'JZZ{k}' for k in range(2,61)]+['BOUNDARY_OUT'];boundaries=list(zip(gatex,codes))
if not np.all(np.diff(gatex)>0):raise ValueError('闸位里程不递增')
# 只需起点、72h切点、终点水位。采用原始yz；按位置找所属渠池，不依据存在拼写误差的POOL名称推渠池。
ks=[0,144,1440];parts=[]
def A(z,yz):
 ans=np.zeros(3)
 for (y0,z0),(y1,z1) in zip(yz[:-1],yz[1:]):
  h0=z-z0;h1=z-z1;w=abs(y1-y0);ans+=np.where((h0>=0)&(h1>=0),(h0+h1)*w/2,0)
  if z1!=z0:ans+=np.where((h0>0)^(h1>0),np.maximum(h0,h1)**2*w/(2*abs(z1-z0)),0)
 return ans
for name,ch in B['channels'].items():
 secs=sorted([B['sections'][s] for s in ch['sec_names']],key=lambda s:s['location']);mid=(secs[0]['location']+secs[-1]['location'])/2;p=int(np.clip(np.searchsorted(gatex,mid)-1,0,len(codes)-2));up,down=codes[p:p+2];x0,x1=gatex[p:p+2]
 if up=='BOUNDARY_IN':zu=obs(up,'闸后水位')[ks];xu=x0
 else:zu=obs(up,'闸后水位')[ks];xu=float(anchor.loc[(up,'闸后'),'anchor_chainage'])
 if down=='BOUNDARY_OUT':zd=obs(down,'闸前水位')[ks];xd=x1
 else:zd=obs(down,'闸前水位')[ks];xd=float(anchor.loc[(down,'闸前'),'anchor_chainage'])
 if xd<=xu:raise ValueError((name,xu,xd))
 for mode in ['原始水位','作者固定偏移水位']:
  cu=.245 if up=='JZZ9' else .301 if up=='JZZ48' else 0.;cd=.245 if down=='JZZ9' else 0.
  uu=zu+(cu if mode!='原始水位' else 0);dd=zd+(cd if mode!='原始水位' else 0);areas=[A(uu+(dd-uu)*(s['location']-xu)/(xd-xu),s['yz']) for s in secs]
  for a,b,aa,ab in zip(secs[:-1],secs[1:],areas[:-1],areas[1:]):
   if b['location']>a['location']:parts.append((mode,a['location'],b['location'],aa,ab))
def volume(lo,hi,mode):
 v=np.zeros(3);coverage=0.
 for md,a,b,aa,ab in parts:
  l=max(lo,a);h=min(hi,b)
  if md!=mode or h<=l:continue
  al=aa+(ab-aa)*(l-a)/(b-a);ah=aa+(ab-aa)*(h-a)/(b-a);v+=(al+ah)*(h-l)/2;coverage+=h-l
 return v,coverage/(hi-lo)
def integrate(v,l,r):return float(np.trapezoid(v[l:r+1],dx=1800))
periods=[('首72h训练段',0,144,0,1),('后27d保留段',144,1440,1,2),('全30天',0,1440,0,2)];records=[]
def balance(label,lo,hi,qa,qb,scope):
 takes=[n for n,x in bleedx.items() if lo<x<=hi];qdiv=sum((bleed[n] for n in takes),start=np.zeros(1441))
 for mode in ['原始水位','作者固定偏移水位']:
  v,cover=volume(lo,hi,mode)
  for period,l,r,a,b in periods:
   dur=(r-l)*1800;vin=integrate(qa,l,r);vout=integrate(qb,l,r);vd=integrate(qdiv,l,r);delta=float(v[b]-v[a]);res=vin-vout-vd-delta
   records.append(dict(scope=scope,interval=label,period=period,level_basis=mode,up_chainage_m=lo,down_chainage_m=hi,duration_s=dur,open_channel_fraction=cover,inflow_m3=vin,outflow_m3=vout,offtake_m3=vd,estimated_open_channel_deltaV_m3=delta,without_storage_residual_m3=vin-vout-vd,residual_m3=res,residual_equiv_m3s=res/dur,mean_inflow_m3s=vin/dur,mean_outflow_m3s=vout/dur,mean_offtake_m3s=vd/dur,storage_equiv_m3s=delta/dur,relative_residual_to_inflow=res/vin,offtake_count=len(takes),offtake_targets=';'.join(takes)))
balance('陶岔—末模型边界（以近末测流代表）',0,xmax,qin,qout,'全线近似')
balance('陶岔—刁河测流断面',0,float(pos.set_index('stationcode').loc['JZZ2','usable_meter_chainage']),qin,obs('JZZ2','瞬时流量'),'首池')
usable=pos[pos.precise_position_usable.astype(str).str.lower().eq('true')].sort_values('usable_meter_chainage')
for (_,a),(_,b) in zip(usable.iloc[:-1].iterrows(),usable.iloc[1:].iterrows()):balance(a.stationcode+'—'+b.stationcode,float(a.usable_meter_chainage),float(b.usable_meter_chainage),obs(a.stationcode,'瞬时流量'),obs(b.stationcode,'瞬时流量'),'可定位测流区间')
# 独立补最后可定位测流站到末测流断面，不混入43个主清单。
a=usable.iloc[-1];balance(a.stationcode+'—北拒马测流断面',float(a.usable_meter_chainage),qxout,obs(a.stationcode,'瞬时流量'),qout,'末段近似')
df=pd.DataFrame(records);df.to_csv(O/'2025水量闭合分段.csv',index=False);df[df.scope.eq('可定位测流区间')&df.period.eq('全30天')].to_csv(O/'2025水量闭合沿程30天.csv',index=False)
major=df[df.scope.isin(['全线近似','首池'])];summary=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),start=str(start),split=str(split),end=str(end),position_usable_stations=len(usable),adjacent_located_intervals=len(usable)-1,total_offtake_nodes=len(bleed),full_line_mean_qin_minus_offtake_m3s=float(np.trapezoid(qin-sum(bleed.values()),dx=1800)/(30*86400)),records=major.to_dict('records'),limits=['ΔV为模型原始yz断面与起末实测水面线插值估算，不是独立量测蓄量。','未建模有压建筑物蓄量变化，明渠覆盖比逐区间列出；内部水位计有未核准安装桩号。','全线末流量计1197369m、末模型断面1197636m，相差267m；末闸前两安装桩号相差1km矛盾保留。','作者固定偏移仅JZZ9前后+0.245m和JZZ48后+0.301m，不是高程复测结论。','43个相邻可定位测流区间可能跨过无法定位的闸站，但仅使用两端可定位测流值。','128分水按C00已补值输入与模型取水交点桩号扣除，不新增未定位9个非零源站，不重复扣取。','闭合残差是边界、取水、测流、水位和模型断面误差的组合，不据此断言仪表错误。'])
(O/'2025水量闭合摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(major[['scope','period','level_basis','residual_equiv_m3s','storage_equiv_m3s','open_channel_fraction']].to_string(index=False));print('usable',len(usable));print(df.query("scope=='可定位测流区间' and period=='全30天' and level_basis=='原始水位'")[['interval','residual_equiv_m3s','storage_equiv_m3s']].to_string(index=False))
def md(d):return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
lines=['# 2025水量闭合诊断','',f'时间{start}至{end}，前72小时与后27天分开。44个位置可用测流站形成43个相邻区间。全线与首池按同一时间积分，正残差表示未解释的蓄水或损失，负残差表示未解释的来水或测流差。','',md(major[['scope','period','level_basis','residual_equiv_m3s','storage_equiv_m3s','open_channel_fraction']]),'','## 沿程残差','',md(df.query("scope=='可定位测流区间' and period=='全30天' and level_basis=='原始水位'")[['interval','residual_equiv_m3s','storage_equiv_m3s']]),'','## 方法边界','']+['- '+s for s in summary['limits']]
(R/'08_报告/2025水量闭合诊断.md').write_text('\n'.join(lines))
# 仅用首72h残差冻结假设，后27d不给参数回填。
raw=df[df.level_basis=='原始水位'].set_index(['interval','period']);train='首72h训练段';hold='后27d保留段';candidates=[]
def rr(name,p):return float(raw.loc[(name,p),'residual_equiv_m3s'])
for station,left,right in [('JZZ2','陶岔—刁河测流断面','JZZ2—JZZ3'),('JZZ23','JZZ22—JZZ23','JZZ23—JZZ24'),('JZZ32','JZZ31—JZZ32','JZZ32—JZZ33')]:
 correction=(rr(left,train)-rr(right,train))/2
 candidates.append(dict(candidate=station+'共同测流端点等效偏移',scope='观测诊断，不改水动力边界',frozen_correction_m3s=correction,train_up_before=rr(left,train),train_down_before=rr(right,train),train_up_after=rr(left,train)-correction,train_down_after=rr(right,train)+correction,hold_up_before=rr(left,hold),hold_down_before=rr(right,hold),hold_up_after=rr(left,hold)-correction,hold_down_after=rr(right,hold)+correction,note='加到该测流观测的诊断偏移；共用两侧闭合约束，仍可能含分水与蓄量/位置误差，不是仪表率定'))
combined_train=rr('陶岔—刁河测流断面',train)+rr('JZZ2—JZZ3',train);combined_hold=rr('陶岔—刁河测流断面',hold)+rr('JZZ2—JZZ3',hold)
candidates.append(dict(candidate='陶岔至湍河合并控制体的入口小偏移假设',scope='如需输入对照，可独立做小幅边界敏感性，不能宣称修正真值',frozen_correction_m3s=-combined_train,train_up_before=combined_train,train_up_after=0.,hold_up_before=combined_hold,hold_up_after=combined_hold-combined_train,note='排除内部刁河测流端点影响后，只剩约0.55m3/s，不能支持入口+5.5；不新增取水、不按水位评分定参数'))
fulltrain=rr('陶岔—末模型边界（以近末测流代表）',train);fullhold=rr('陶岔—末模型边界（以近末测流代表）',hold)
candidates.append(dict(candidate='全线训练残差直接归入口（否决）',scope='不建议转为仿真输入',frozen_correction_m3s=-fulltrain,train_up_before=fulltrain,train_up_after=0.,hold_up_before=fullhold,hold_up_after=fullhold-fulltrain,note='首72h和保留期差异很大；全线明渠蓄量估算及未定位源影响尚未闭合'))
pd.DataFrame(candidates).to_csv(O/'2025水量闭合冻结候选与保留验证.csv',index=False)
summary['frozen_candidates']=candidates;summary['recommendation']='优先检查JZZ2、JZZ23、JZZ32共同测流端点的双侧反号残差；如一定要做水动力输入对照，候选仅为陶岔−0.552884m3/s量级的小幅固定偏移，由首72h陶岔至湍河联合水量冻结。不得将首池−5.5直接补到入口；不采用全线训练残差+12.48的入口修正。';(O/'2025水量闭合摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines+=['','## 后续对照优先级','',summary['recommendation'],'','首池与下一渠段的反号残差在首72小时、后27天都稳定出现，合并后相抵。因此先查共同测流端点，不能将一个渠段的残差归给陶岔边界或补成未定位取水。JZZ23和JZZ32也有类似反号组合。','',md(pd.DataFrame(candidates).fillna('')),'','小幅入口假设只是水量控制体约束下的敏感性分支，不是已经证明的修正。全线没有单向、恒定的沿程残差，未定位分水也不能凭水量差擅自入模。']
(R/'08_报告/2025水量闭合诊断.md').write_text('\n'.join(lines))
