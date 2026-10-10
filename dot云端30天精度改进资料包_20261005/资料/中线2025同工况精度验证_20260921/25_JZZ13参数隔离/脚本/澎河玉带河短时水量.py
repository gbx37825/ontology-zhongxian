"""只读水量闭合；不率定、不改节点，不以水位评分反调参数。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[2];I=R/'02_输入';O=R/'07_对比分析';P=R/'03_运行参数'
path=R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json';M=json.loads(path.read_text())['中线全线'];B=M['baseData'];start=pd.Timestamp('2025-07-22 12:00');end=pd.Timestamp('2025-08-21 12:00');split=start+pd.Timedelta(hours=72);idx=pd.date_range(start,end,freq='30min');F=pd.read_csv(I/'2025节制闸原始窗口.csv',parse_dates=['时间']);mapping=pd.read_csv(I/'2025节制闸站名映射.csv').set_index('station_code');pos=pd.read_csv(I/'优先级确认后测流映射.csv');gatepos=pd.read_csv(I/'测点位置映射.csv').set_index('stationcode');anchor=pd.read_csv(R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/2019水量闭合水位锚点.csv').set_index(['station','side'])
def obs(code,col):return F[F.站点名称==mapping.loc[code,'source_name']].set_index('时间').loc[idx,col].to_numpy(float)
qin=obs('BOUNDARY_IN','瞬时流量');qout=obs('BOUNDARY_OUT','瞬时流量');bleed={n:np.array(M['timeSeries']['seriesData'][s],float)[:,1] for s,(t,n) in M['timeSeries']['seriesLink'].items() if t=='bleeders'}
bleedx={n:float(B['bleeders'][n]['channel'][1]) for n in bleed};xmax=max(s['location'] for s in B['sections'].values());qxout=1197369. # 附件791行末流量计，不与末水位1197636混同。
# 两端差额分开：全模型边界以近末端流量计代出流含267m位置差；43个相邻已知测流区间用实际测流桩号。
gatex=[0.]+[float(gatepos.loc[f'JZZ{k}','gate_chainage']) for k in range(2,61)]+[xmax];codes=['BOUNDARY_IN']+[f'JZZ{k}' for k in range(2,61)]+['BOUNDARY_OUT'];boundaries=list(zip(gatex,codes))
if not np.all(np.diff(gatex)>0):raise ValueError('闸位里程不递增')
# 只需起点、72h切点、终点水位。采用原始yz；按位置找所属渠池，不依据存在拼写误差的POOL名称推渠池。
ks=[0,12,48];parts=[]
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

records=[]
for up,down in [('JZZ12','JZZ14')]:
 lo=float(pos.set_index('stationcode').loc[up,'usable_meter_chainage']);hi=float(pos.set_index('stationcode').loc[down,'usable_meter_chainage'])
 takes=[n for n,x in bleedx.items() if lo<x<=hi];qd=sum((bleed[n] for n in takes),start=np.zeros(1441));v,coverage=volume(lo,hi,'原始水位')
 for label,l,r,a,b in [('首6小时',0,12,0,1),('第6至24小时',12,48,1,2),('首24小时',0,48,0,2)]:
  dur=(r-l)*1800;vi=integrate(obs(up,'瞬时流量'),l,r);vo=integrate(obs(down,'瞬时流量'),l,r);vd=integrate(qd,l,r);dv=float(v[b]-v[a]);records.append(dict(interval=up+'—'+down,period=label,inflow_mean=vi/dur,outflow_mean=vo/dur,offtake_mean=vd/dur,estimated_storage_change=dv,estimated_storage_rate=dv/dur,residual_equiv_m3s=(vi-vo-vd-dv)/dur,channel_coverage=coverage,offtake_targets=takes))
O=R/'25_JZZ13参数隔离';(O/'澎河至玉带河短时水量核查.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False,indent=2))
for name,ch in B['channels'].items():
 if name.lower().startswith(('pool11-','pool12-','pool13-','pool14-')) or name=='pool3-2-DHX15DHX16':
  sx=sorted(B['sections'][n]['location'] for n in ch['sec_names']);print(name,sx[0],sx[-1])
