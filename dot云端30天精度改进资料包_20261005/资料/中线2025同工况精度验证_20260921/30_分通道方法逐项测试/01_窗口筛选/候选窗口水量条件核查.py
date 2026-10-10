"""原断面与水位插值估算蓄变，不将水量残差当仪表或边界修正值。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];I=R/'26_冻结基线与后续归因/共享输入'
M=json.loads((I/'C07.json').read_text())['中线全线'];B=M['baseData'];T=np.arange(1441)*1800;ob=pd.read_csv(I/'作者已核实修正观测层.csv');pos=pd.read_csv(R/'02_输入/测点位置映射.csv').set_index('stationcode');anchor=pd.read_csv(R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/2019水量闭合水位锚点.csv').set_index(['station','side'])
Z={}
for (station,metric),g in ob.groupby(['station_code','metric']):
 g=g.set_index('time_s').reindex(T);Z[station,metric]=g.value.where(g.valid.fillna(False)).to_numpy(float)
bleed={n:np.asarray(M['timeSeries']['seriesData'][sid],float)[:,1] for sid,(kind,n) in M['timeSeries']['seriesLink'].items() if kind=='bleeders'};bleedx={n:float(B['bleeders'][n]['channel'][1]) for n in bleed}
gatex=[0]+[float(pos.loc[f'JZZ{k}','gate_chainage']) for k in range(2,61)]+[max(s['location'] for s in B['sections'].values())];codes=['BOUNDARY_IN']+[f'JZZ{k}' for k in range(2,61)]+['BOUNDARY_OUT'];rows=[]
def area(z,yz):
 ans=np.zeros(len(z))
 for (y0,z0),(y1,z1) in zip(yz[:-1],yz[1:]):
  h0=z-z0;h1=z-z1;w=abs(y1-y0);ans+=np.where((h0>=0)&(h1>=0),(h0+h1)*w/2,0)
  if z1!=z0:ans+=np.where((h0>0)^(h1>0),np.maximum(h0,h1)**2*w/(2*abs(z1-z0)),0)
 return ans

def volume(lo,hi,ix):
 total=np.zeros(len(ix));coverage=0.;used=[]
 for name,ch in B['channels'].items():
  secs=sorted([B['sections'][n] for n in ch['sec_names']],key=lambda s:s['location'])
  if secs[-1]['location']<=lo or secs[0]['location']>=hi:continue
  mid=(secs[0]['location']+secs[-1]['location'])/2;p=int(np.clip(np.searchsorted(gatex,mid)-1,0,len(codes)-2));u,d=codes[p:p+2]
  if 'BOUNDARY' in u or 'BOUNDARY' in d:raise ValueError('候选区间不应触及边界')
  zu=Z[u,'afterbay_m'][ix];zd=Z[d,'forebay_m'][ix];xu=float(anchor.loc[(u,'闸后'),'anchor_chainage']);xd=float(anchor.loc[(d,'闸前'),'anchor_chainage']);assert xd>xu
  aa=[area(zu+(zd-zu)*(s['location']-xu)/(xd-xu),s['yz']) for s in secs]
  for a,b,va,vb in zip(secs[:-1],secs[1:],aa[:-1],aa[1:]):
   l=max(lo,a['location']);h=min(hi,b['location']);length=b['location']-a['location']
   if h<=l:continue
   al=va+(vb-va)*(l-a['location'])/length;ah=va+(vb-va)*(h-a['location'])/length;total+=(al+ah)*(h-l)/2;coverage+=h-l
  used.append([u,d])
 return total,coverage/(hi-lo),used
for row in json.loads((O/'候选窗口摘要.json').read_text())['selected']:
 station=row['station'];station_n=int(station[3:]);l=int(row['start_s']/1800);r=int(row['end_s']/1800);dur=(r-l)*1800
 usable=pos[(pos.status=='已核对')&pos.meter_chainage.notna()].sort_values('meter_chainage');names=usable.index.tolist();idx=names.index(station)
 for u,d in [(names[idx-1],station),(station,names[idx+1])]:
  lo=float(pos.loc[u,'meter_chainage']);hi=float(pos.loc[d,'meter_chainage']);takes=[n for n,x in bleedx.items() if lo<x<=hi];qb=sum((bleed[n] for n in takes),start=np.zeros(len(T)));qin=Z[u,'flow_m3s'];qout=Z[d,'flow_m3s'];v,cov,used=volume(lo,hi,[l,r]);dv=float(v[1]-v[0]);a,b,c=[float(np.trapezoid(x[l:r+1],dx=1800)/dur) for x in [qin,qout,qb]]
  rows.append(dict(station=station,start_s=row['start_s'],end_s=row['end_s'],interval=u+'—'+d,up_meter_m=lo,down_meter_m=hi,mean_inflow_m3s=a,mean_outflow_m3s=b,mean_offtake_m3s=c,without_storage_residual_m3s=a-b-c,estimated_deltaV_m3=dv,estimated_storage_m3s=dv/dur,conditional_balance_residual_m3s=a-b-c-dv/dur,conditional_relative_residual=(a-b-c-dv/dur)/a,open_channel_coverage=cov,offtake_targets=';'.join(takes),water_level_anchor_method='既有2019安装锚点+2025作者确认水位；不是新核准安装位置'))
df=pd.DataFrame(rows);df.to_csv(O/'候选窗口水量条件核查.csv',index=False);print(df[['station','interval','mean_inflow_m3s','mean_outflow_m3s','mean_offtake_m3s','estimated_storage_m3s','conditional_balance_residual_m3s','open_channel_coverage']].to_string(index=False))
(O/'水量核查边界.json').write_text(json.dumps({'limits':['ΔV用原始断面和实测水位沿桩号线性插值估算，沿用现有安装锚点；不是独立实测蓄量，不证明现场真值准确。','有压建筑物及短结构蓄量未独立重建，覆盖比例只计原始明渠断面范围。','采用C07分水序列，原始填补和假设仍保留，不因差額新增流量。','准稳态时间波动小不等于控制体水量闭合；残差可能混有流量、分水、蓄量、位置及几何误差。','JZZ16扩展上游JZZ15测流桩号未闭合，因此跨至可定位的JZZ14核查，不冒充JZZ15—JZZ16精确水量。']},ensure_ascii=False,indent=2))
