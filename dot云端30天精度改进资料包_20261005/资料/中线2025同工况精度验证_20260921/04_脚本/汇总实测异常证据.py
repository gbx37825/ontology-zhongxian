from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=R/'12_全量实测质量核查'
e=pd.read_csv(O/'开度突变逐事件.csv');raw=pd.read_csv(O/'节制闸_全量原值.csv.gz',parse_dates=['时间']);q=pd.read_csv(O/'全站全字段质量统计.csv');cov=pd.read_csv(O/'时间覆盖与缺测统计.csv');caps=pd.read_csv(O/'闸孔台账与全年开度核对.csv')
for c in raw:
 if '开度' in c:raw[c]=pd.to_numeric(raw[c],errors='coerce')/1000
extra=[]
for name,evs in e.groupby('station'):
 g=raw[raw.站点名称==name].sort_values('时间').reset_index(drop=True)
 for ix,ev in evs.iterrows():
  i=int(np.flatnonzero(g.源行号.to_numpy()==ev.source_row)[0]);row={'event_index':ix};t=g.时间.iloc[i]
  for label,a,b in [('core',max(0,i-1),min(len(g),i+2)),('lag2h',max(0,i-1),int(g.时间.searchsorted(t+pd.Timedelta(hours=2),side='right'))),('lag4h',max(0,i-1),int(g.时间.searchsorted(t+pd.Timedelta(hours=4),side='right')))]:
   for c,tag in [('闸前水位','hu'),('闸后水位','hd'),('瞬时流量','q')]:
    v=g[c].iloc[a:b].to_numpy(float);row[label+'_'+tag+'_range']=float(v.max()-v.min()) if np.isfinite(v).all() and (v>=0).all() else np.nan
  extra.append(row)
ex=pd.DataFrame(extra).set_index('event_index');e=e.join(ex);e.to_csv(O/'开度突变含延迟响应核查.csv',index=False)
strict=e.classification.str.startswith('目标平稳');st=e[strict]
counts=st.groupby(['station','code'],dropna=False).agg(events=('hole','size'),short_pulses=('event_type',lambda x:int((x=='2小时内回归的短脉冲').sum())),author_30d=('in_author_30d','sum')).sort_values('events',ascending=False);counts.to_csv(O/'疑似异常站点排序.csv')
sensitivity=[]
for tol in [.03,.05,.1]:
 ok=(e.hu_range_m<=tol)&(e.hd_range_m<=tol)&(e.q_range_m3s<=e.q_threshold_m3s)&(e.target_range_m<=.1)&(abs(e.peak_m-e.target_at_peak)>=1)
 sensitivity.append(dict(water_threshold_m=tol,events=int(ok.sum()),stations=int(e[ok].station.nunique())))
# “空白”可能是不存在的第4孔，“负水深”可能是仪表零点；分开列原始事实，不一律判坏。
interesting=q[(q.sentinel_le_minus100>0)|(q.longest_identical_nonzero_points*q.modal_step_s>=86400)].copy();interesting.to_csv(O/'负码与长时间恒值候选.csv',index=False)
summary=dict(coverage=cov.groupby('kind').agg(stations=('station','size'),rows=('rows','sum'),start=('start','min'),end=('end','max'),gaps=('gaps_gt_1p5step','sum'),duplicate_rows=('duplicate_rows','sum')).reset_index().to_dict('records'),event_total=len(e),event_stations=e.station.nunique(),event_short=int((e.event_type=='2小时内回归的短脉冲').sum()),strict_events=len(st),strict_stations=st.station.nunique(),strict_model_gate_events=int(st.code.fillna('').str.startswith('JZZ').sum()),strict_boundary_in_events=int((st.code=='BOUNDARY_IN').sum()),author30_events=int(e.in_author_30d.sum()),author30_strict=int(st.in_author_30d.sum()),sensitivity=sensitivity,strict_still_quiet_after_2h=int(((st.lag2h_hu_range<=.05)&(st.lag2h_hd_range<=.05)&(st.lag2h_q_range<=st.q_threshold_m3s)).sum()),strict_still_quiet_after_4h=int(((st.lag4h_hu_range<=.05)&(st.lag4h_hd_range<=.05)&(st.lag4h_q_range<=st.q_threshold_m3s)).sum()),cap_mapped_holes=int(caps.cap_m.notna().sum()),over_cap_0p1m_points=int(caps.over_cap_0p1m.fillna(0).sum()),water_flow_isolated_candidates=len(pd.read_csv(O/'水位流量孤立跳点.csv')),quality_negative_cell_counts=q.groupby('kind').negative.sum().to_dict(),quality_sentinel_cell_counts=q.groupby('kind').sentinel_le_minus100.sum().to_dict(),source_unchanged=True,thresholds={'opening_jump_m':1,'return_tolerance_m':.1,'short_duration_h':2,'target_range_m':.1,'hydraulic_water_range_m':.05,'flow_range_threshold':'max(5m3/s,5%medianQ)'},scope_exclusions=['渠道水体截图ZIP和逐月蓄水PDF未逐图OCR，不作为已完成的时序逐点筛查','机组转速表单独检查原值，混合日期及累计日时间未强行映射到泵站时刻'],rules_are_heuristics_not_sensor_fault_proof=True)
(O/'全量核查摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str));print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
font=Path('/System/Library/Fonts/STHeiti Light.ttc');font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name();plt.rcParams['axes.unicode_minus']=False
cases=[('七里河倒虹吸出口节制闸',pd.Timestamp('2025-07-22 12:00'),pd.Timestamp('2025-07-22 22:00'),3,'七里河两次开度脉冲')]
for code in ['JZZ27','JZZ23']:
 sel=st[(st.code==code)&st.in_author_30d].sort_values('amplitude_m',ascending=False)
 if len(sel):ev=sel.iloc[0];t=pd.Timestamp(ev.peak_time);cases.append((ev.station,t-pd.Timedelta(hours=3),t+pd.Timedelta(hours=4),int(ev.hole),code+'疑似异常示例'))
selected=[]
for num,(name,lo,hi,hole,title) in enumerate(cases,1):
 g=raw[(raw.站点名称==name)&(raw.时间>=lo)&(raw.时间<=hi)].copy();g.to_csv(O/f'示例{num}_原值开度米.csv',index=False);fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True,constrained_layout=True);x=g.时间
 ax[0].plot(x,g[f'孔{hole}实际开度'],'o-',label=f'{hole}孔实际');ax[0].plot(x,g[f'孔{hole}目标开度'],'--',label=f'{hole}孔目标');ax[0].set_ylabel('开度(m)');ax[0].set_title(title+'：'+name+'（全部为实测原值，开度mm换算m）');ax[0].legend()
 ax[1].plot(x,g.闸前水位,'o-',label='闸前');ax[1].plot(x,g.闸后水位,'o-',label='闸后');ax[1].set_ylabel('水位(m)');ax[1].legend()
 ax[2].plot(x,g.瞬时流量,'o-',color='#428353');ax[2].set_ylabel('实测流量(m³/s)')
 for k in range(1,5):
  if k!=hole and g[f'孔{k}实际开度'].notna().any():ax[3].plot(x,g[f'孔{k}实际开度'],label=f'{k}孔实际')
 ax[3].set_ylabel('其他孔开度(m)');ax[3].legend();ax[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))
 for a in ax:a.grid(alpha=.2)
 fig.savefig(O/f'示例{num}_开度水位流量对照.png',dpi=200);fig.savefig(O/f'示例{num}_开度水位流量对照.svg');plt.close(fig);selected.append(dict(station=name,hole=hole,start=str(lo),end=str(hi),plot=f'示例{num}_开度水位流量对照.png'))
(O/'重点示例.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
# 删除本轮未换算开度单位的初次调试派生输出，原始压缩CSV不动。
for name in ['开度短脉冲逐事件.csv','五个重点站30天原值.csv.gz']:
 p=O/name
 if p.exists():p.unlink()
