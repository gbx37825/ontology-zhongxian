from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=R/'13_正常调闸响应';normal=pd.read_csv(O/'正常调闸逐事件响应.csv');stats=pd.read_csv(O/'同站分幅度分方向响应统计.csv');raw=pd.read_csv(R/'12_全量实测质量核查/节制闸_全量原值.csv.gz',parse_dates=['时间']);events=pd.read_csv(R/'12_全量实测质量核查/开度突变含延迟响应核查.csv');maps=json.loads((R/'02_输入/2025节制闸站名映射.json').read_text());maps={x['station_code']:x for x in maps if x['station_code'].startswith('JZZ')};model=next(iter(json.loads((R/'02_输入/C00_实际开度负值修复_2025同工况30天.json').read_text()).values()))
result=[]
for code,evs in events[events.code.fillna('').str.startswith('JZZ')].groupby('code'):
 meta=maps[code];holes=list(map(int,meta['model_holes'].split(';')));width=np.array([next(v['b'] for n,v in model['baseData']['gates'].items() if n.startswith(code+'-') and n.endswith('-'+str(k))) for k in holes]);g=raw[raw.站点名称==meta['source_name']].sort_values('时间').reset_index(drop=True);a=g[[f'孔{k}实际开度' for k in holes]].to_numpy(float)/1000;h=g[['闸前水位','闸后水位','瞬时流量']].to_numpy(float);indices={v:i for i,v in enumerate(g.源行号)}
 for ev in evs.itertuples():
  i=indices[ev.source_row];i0=int(g.时间.searchsorted(pd.Timestamp(ev.start)));before=a[max(0,i0-2):i0].mean(axis=0);delta=a[i]-before;da=float(delta@width/width.sum());bh=h[max(0,i0-2):i0].mean(axis=0);obs=h[i:i+2].mean(axis=0)-bh;direction='开大' if da>0 else '关小';ref=normal[(normal.code==code)&(normal.direction==direction)&(normal.lag_h==.5)]
  amp=ref[(abs(ref.delta_mean_opening_m)>=.5*abs(da))&(abs(ref.delta_mean_opening_m)<=2*abs(da))]
  near=amp[(abs(amp.base_q-bh[2])<=max(5,.3*abs(bh[2])))&(abs((amp.base_hu-amp.base_hd)-(bh[0]-bh[1]))<=max(.1,.3*abs(bh[0]-bh[1])))&(abs(amp.base_mean_opening_m-before@width/width.sum())<=max(.2,.5*abs(before@width/width.sum())))]
  pre_stable=np.isfinite(a[max(0,i0-2):i0]).all() and np.max(np.ptp(a[max(0,i0-2):i0],axis=0))<=.02
  trajectory=ev.event_type!='2小时内回归的短脉冲' or (pd.Timestamp(ev.end)-pd.Timestamp(ev.start)).total_seconds()>1800
  mixed=bool((delta>.02).any() and (delta<-.02).any())
  reason='存在同站同量级相近工况样本' if len(near)>=10 else '相近工况样本不足10个'
  if not pre_stable:reason+='；调前开度不稳'
  if not trajectory:reason+='；短脉冲与持续调闸持续时间不同'
  if mixed:reason+='；其他孔反向动作'
  result.append(dict(code=code,station=ev.station,hole=ev.hole,peak_time=ev.peak_time,event_type=ev.event_type,delta_station_mean_m=da,station_normal_same_direction_max_m=float(abs(ref.delta_mean_opening_m).max()) if len(ref) else np.nan,same_amplitude_count=len(amp),matched_count=len(near),pre_stable=pre_stable,trajectory_comparable=trajectory,mixed_action=mixed,observed_hu_m=obs[0],observed_hd_m=obs[1],observed_q_m3s=obs[2],assessment=reason))
pd.DataFrame(result).to_csv(O/'疑似事件正常样本支持度.csv',index=False)
# 主文表：同站、同方向、同幅度，1h时间窗口。展示平均与样本数，范围另列。
sel=stats[(stats.scope=='同站正常调闸')&(stats.code.isin(['JZZ42','JZZ27']))&(stats.lag_h==1)&(stats.amplitude_bin.isin(['0.05–0.10','0.10–0.25']))]
sel.to_csv(O/'重点站正常调节响应表.csv',index=False)
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for row,code in enumerate(['JZZ42','JZZ27']):
 d=normal[(normal.code==code)&(normal.lag_h==1)]
 for col,(field,label) in enumerate([('delta_hu_m','闸前水位变化(cm)'),('delta_hd_m','闸后水位变化(cm)')]):
  ax=axs[row,col];ax.scatter(d.delta_mean_opening_m*100,d[field]*100,s=10,alpha=.4);ss=stats[(stats.scope=='同站正常调闸')&(stats.code==code)&(stats.lag_h==1)&(stats['count']>=5)]
  ax.errorbar(ss.mean_opening_change_m*100,ss[field+'_median']*100,yerr=np.vstack([(ss[field+'_median']-ss[field+'_p10'])*100,(ss[field+'_p90']-ss[field+'_median'])*100]),fmt='o',color='#c54b26',capsize=3,label='分组中位数及10%—90%范围');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set(title=code+' 正常调闸1小时响应',xlabel='全站孔宽加权平均开度变化(cm)',ylabel=label);ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.savefig(O/'正常调闸幅度与水位响应.png',dpi=210);fig.savefig(O/'正常调闸幅度与水位响应.svg');plt.close(fig)
print(sel[['code','direction','count','mean_opening_change_m','delta_hu_m_mean','delta_hd_m_mean','delta_q_m3s_mean']].to_string(index=False));print(pd.DataFrame(result).query("code=='JZZ42' and peak_time.str.startswith('2025-07-22')",engine='python').to_string(index=False))
