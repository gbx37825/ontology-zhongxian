"""用命令与反馈筛正常调闸，不按水位响应大小挑样本；统计事件响应，不作因果回归外推。"""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'13_正常调闸响应';O.mkdir(exist_ok=True)
df=pd.read_csv(R/'12_全量实测质量核查/节制闸_全量原值.csv.gz',parse_dates=['时间']);mapping=json.loads((R/'02_输入/2025节制闸站名映射.json').read_text());source=next(iter(json.loads((R/'02_输入/C00_实际开度负值修复_2025同工况30天.json').read_text()).values()))
groups={};lookup={}
for meta in mapping:
 code=meta['station_code']
 if not code.startswith('JZZ'):continue
 name=meta['source_name'];g=df[df.站点名称==name].sort_values('时间').reset_index(drop=True);holes=list(map(int,meta['model_holes'].split(';')));width=np.array([next(v['b'] for n,v in source['baseData']['gates'].items() if n.startswith(code+'-') and n.endswith('-'+str(k))) for k in holes]);a=g[[f'孔{k}实际开度' for k in holes]].to_numpy(float)/1000;t=g[[f'孔{k}目标开度' for k in holes]].to_numpy(float)/1000;h=g[['闸前水位','闸后水位','瞬时流量']].to_numpy(float);groups[code]=(g,a,t,h,width);lookup[code]={v:i for i,v in enumerate(g.时间)}
rows=[];candidates=[]
for code,(g,a,t,h,w) in groups.items():
 n=len(g);mean=a@w/w.sum();targetmean=t@w/w.sum();times=g.时间
 change=np.max(abs(np.diff(t,axis=0)),axis=1);inds=np.flatnonzero(change>=.03)+1
 for i in inds:
  if i<3 or i+2>=n:continue
  pre=t[i-2:i];delta=t[i]-np.mean(pre,axis=0);dm=float(delta@w/w.sum())
  reason='通过'
  if not np.isfinite(pre).all() or (pre<0).any() or np.max(np.ptp(pre,axis=0))>.02:reason='调前目标不稳定或无效'
  elif abs(dm)<.02:reason='总开口变化过小或各孔抵消'
  elif (delta>.02).any() and (delta<-.02).any():reason='多孔相反动作'
  elif not np.isfinite(a[i-2:i+2]).all() or (a[i-2:i+2]<0).any() or np.max(abs(a[i-2:i+2]-t[i-2:i+2]))>.05:reason='实际未跟随目标或反馈异常'
  candidates.append(dict(code=code,time=str(times.iloc[i]),reason=reason))
  if reason!='通过':continue
  for lag in [0.5,1,2,4]:
   end=i+int(lag*2)
   if end>=n:continue
   if ((times.iloc[i-2:end+1].diff().dt.total_seconds().dropna()!=1800).any()):continue
   if np.max(np.ptp(t[i:end+1],axis=0))>.02:continue
   if not np.isfinite(a[i-2:end+1]).all() or (a[i-2:end+1]<0).any() or np.max(abs(a[i-2:end+1]-t[i-2:end+1]))>.05:continue
   hh=h[i-2:end+1]
   if not np.isfinite(hh).all() or (hh[:,:2]<=0).any() or (hh[:,2]<0).any():continue
   neighbor=True;neighbor_known=0
   for num in [int(code[3:])-1,int(code[3:])+1]:
    nc='JZZ'+str(num)
    if nc not in groups:continue
    ng,na,nt,nh,nw=groups[nc];idx=lookup[nc].get(times.iloc[i])
    if idx is None or idx<2 or idx+int(lag*2)>=len(ng):neighbor=False;continue
    seg=nt[idx-2:idx+int(lag*2)+1];neighbor_known+=1
    if not np.isfinite(seg).all() or (seg<0).any() or np.max(np.ptp(seg,axis=0))>.02:neighbor=False
   before=np.mean(h[i-2:i],axis=0);after=np.mean(h[end-1:end+1],axis=0);resp=after-before;trend=h[i-1]-h[i-2];adj=resp-trend*(lag*2+1)
   da=float(np.mean(mean[i:i+2])-np.mean(mean[i-2:i]));area=da*w.sum()
   if da*dm<=0 or abs(da)<.02 or abs(da-dm)>max(.02,.25*abs(dm)):continue
   rows.append(dict(code=code,station=g.站点名称.iloc[i],time=str(times.iloc[i]),source_row=int(g.源行号.iloc[i]),direction='开大' if da>0 else '关小',lag_h=lag,delta_mean_opening_m=da,delta_area_m2=area,base_mean_opening_m=float(np.mean(mean[i-2:i])),base_hu=before[0],base_hd=before[1],base_q=before[2],delta_hu_m=resp[0],delta_hd_m=resp[1],delta_q_m3s=resp[2],detrended_hu_m=adj[0],detrended_hd_m=adj[1],detrended_q_m3s=adj[2],pretrend_hu_halfhour=trend[0],pretrend_hd_halfhour=trend[1],neighbor_quiet=neighbor,neighbor_known=neighbor_known,all_holes=len(w)))
 print(code,'accepted horizon records',len(rows),flush=True)
r=pd.DataFrame(rows);r['amplitude_bin']=pd.cut(abs(r.delta_mean_opening_m),[0,.05,.1,.25,.5,1,np.inf],labels=['0.02–0.05','0.05–0.10','0.10–0.25','0.25–0.50','0.50–1.00','>1.00'],include_lowest=True);r.to_csv(O/'正常调闸逐事件响应.csv',index=False);pd.DataFrame(candidates).to_csv(O/'目标调闸筛选记录.csv',index=False)
stats=[]
for scope,d in [('同站正常调闸',r),('相邻闸目标平稳',r[r.neighbor_quiet])]:
 for (code,direction,lag,amp),g in d.groupby(['code','direction','lag_h','amplitude_bin'],observed=True):
  v=dict(scope=scope,code=code,direction=direction,lag_h=lag,amplitude_bin=str(amp),count=len(g),mean_opening_change_m=float(g.delta_mean_opening_m.mean()))
  for c in ['delta_hu_m','delta_hd_m','delta_q_m3s']:
   v[c+'_mean']=float(g[c].mean());v[c+'_median']=float(g[c].median());v[c+'_p10']=float(g[c].quantile(.1));v[c+'_p90']=float(g[c].quantile(.9))
  stats.append(v)
pd.DataFrame(stats).to_csv(O/'同站分幅度分方向响应统计.csv',index=False)
summary=dict(records=len(r),unique_events=len(r[['code','time']].drop_duplicates()),stations=r.code.nunique(),neighbor_quiet_events=len(r[r.neighbor_quiet][['code','time']].drop_duplicates()),criteria='目标变化>=0.03m；孔宽加权平均变化>=0.02m；实际与目标偏差<=0.05m；调前1h、调后各响应期目标稳定<=0.02m；不按水位响应大小筛正常；同站分方向/幅度/时滞；相邻闸目标平稳单列')
(O/'提取摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary)
