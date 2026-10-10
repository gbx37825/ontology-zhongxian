"""源表只读审计。规则仅标疑点，不改数据，不把无水力响应等同故障定论。"""
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd,openpyxl
R=Path(__file__).resolve().parents[1];O=R/'12_全量实测质量核查';P=Path('/Users/gaobingxiang/code/十四五/节制闸+分水闸/64座节制闸+54座退水闸+97个分水口 +61个控制闸(2023-4).xlsx')
w=openpyxl.load_workbook(P,read_only=True,data_only=True);registry=[]
for ri,v in enumerate(w['汇总'].values,1):
 if ri<3 or not v[1]:continue
 raw=v[22] if len(v)>22 else None
 nums=re.findall(r'\d+(?:\.\d+)?',str(raw)) if raw is not None else []
 # 显式mm表头；只解析纯数值/斜杠列表，含其他文字的留待核对。
 valid=raw is not None and re.fullmatch(r'[\d.\s/、,，mM毫米]+',str(raw)) is not None
 registry.append(dict(name=str(v[1]).strip(),source_row=ri,holes=v[4],max_opening_raw=raw,max_openings_m=[float(x)/1000 for x in nums] if valid else [],historical_max_raw=v[21],locked_raw=v[23]))
(O/'2023台账开度原值.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2,default=str));reg={d['name']:d for d in registry}
maps=json.loads((R/'02_输入/2025节制闸站名映射.json').read_text());mapping={x['source_name']:x for x in maps}
quality=[];coverage=[];events=[];limits=[];water_spikes=[];contexts=[];duplicates=[]
def rng(a):
 a=np.asarray(a,float);return float(np.nanmax(a)-np.nanmin(a)) if np.isfinite(a).all() and len(a) else np.nan
def basic(df,kind,namecol):
 numeric=[c for c in df if c not in ['源工作表','源行号',namecol,'时间','站点编码','闸站编码']]
 for c in numeric:df[c]=pd.to_numeric(df[c],errors='coerce')
 for name,g in df.groupby(namecol,sort=False):
  g=g.sort_values(['时间','源工作表','源行号']);dup=g.duplicated('时间',keep=False)
  if dup.any():
   for t,dd in g[dup].groupby('时间'):duplicates.append(dict(kind=kind,station=name,time=t,rows=';'.join(dd['源工作表'].astype(str)+':'+dd['源行号'].astype(str)),conflicting=bool((dd[numeric].nunique(dropna=False)>1).any())))
  # 重复时刻不作为独立时间步；保留第一条，仅供筛查，重复另行披露。
  g=g.drop_duplicates('时间').reset_index(drop=True);dt=g['时间'].diff().dt.total_seconds();mode=float(dt[dt>0].mode().iloc[0]) if (dt>0).any() else np.nan
  coverage.append(dict(kind=kind,station=name,rows=len(g),raw_rows=len(g)+int(dup.sum())-int(g['时间'].isin(df.loc[(df[namecol]==name)&df.duplicated([namecol,'时间'],keep=False),'时间']).sum()) if dup.any() else len(g),start=str(g.时间.min()),end=str(g.时间.max()),modal_step_s=mode,gaps_gt_1p5step=int((dt>1.5*mode).sum()),duplicate_rows=int(dup.sum())))
  for c in numeric:
   a=g[c].to_numpy(float);finite=np.isfinite(a);runs=pd.Series(a).ne(pd.Series(a).shift()).cumsum();lens=pd.Series(a).groupby(runs).agg(['first','size']);validruns=lens[(lens['first'].notna())&(lens['first']!=0)];longest=int(validruns['size'].max()) if len(validruns) else 0
   quality.append(dict(kind=kind,station=name,field=c,unit='m' if ('开度' in c or '水位' in c or '水深' in c) else 'm3/s' if '流量' in c else '源单位',total=len(a),missing=int(np.isnan(a).sum()),negative=int((a<0).sum()),sentinel_le_minus100=int((a<=-100).sum()),zero=int((a==0).sum()),min=float(np.nanmin(a)) if finite.any() else None,max=float(np.nanmax(a)) if finite.any() else None,longest_identical_nonzero_points=longest,modal_step_s=mode))
  yield name,g,mode,numeric

def gates():
 df=pd.read_csv(O/'节制闸_全量原值.csv.gz',parse_dates=['时间']);nraw=len(df)
 # 源表开度单位mm，既有输入构造也是/1000；分析派生层换算m，原压缩CSV不动。
 for col in df.columns:
  if '开度' in col:df[col]=pd.to_numeric(df[col],errors='coerce')/1000.
 for name,g,step,numeric in basic(df,'节制闸','站点名称'):
  meta=mapping.get(name,{});code=meta.get('station_code','');modelname=meta.get('model_name',name);rr=reg.get(name,reg.get(modelname));count=int(rr['holes']) if rr and str(rr['holes']).replace('.','',1).isdigit() else None
  caps=rr['max_openings_m'] if rr else [];ts=g.时间.to_numpy();hu=g['闸前水位'].to_numpy();hd=g['闸后水位'].to_numpy();q=g['瞬时流量'].to_numpy();n=len(g)
  for hole in range(1,5):
   c=f'孔{hole}实际开度';tc=f'孔{hole}目标开度';a=g[c].to_numpy();target=g[tc].to_numpy();cap=(caps[hole-1] if len(caps)>1 and len(caps)>=hole else caps[0] if len(caps)==1 else np.nan)
   if count and hole>count and np.isfinite(a).any():limits.append(dict(station=name,code=code,hole=hole,status='台账孔数之外仍有实际值',count=int(np.isfinite(a).sum())))
   valid=np.isfinite(a)&(a>=0);limits.append(dict(station=name,code=code,hole=hole,status='台账比对',ledger_row=None if not rr else rr['source_row'],cap_m=cap,max_actual=float(np.nanmax(a)) if valid.any() else None,over_cap=int((valid&(a>cap)).sum()) if np.isfinite(cap) else None,over_cap_0p1m=int((valid&(a>cap+.1)).sum()) if np.isfinite(cap) else None,actual_target_diff_ge1m=int((valid&np.isfinite(target)&(target>=0)&(abs(a-target)>=1)).sum())))
   # 1—4点、最长2小时的起跳后回落；基线返回误差<=0.10m；起跳>=1m；仅连续采样。
   dif=np.diff(a);starts=np.flatnonzero(np.isfinite(dif)&(abs(dif)>=1))+1;lastend=-1
   for i in starts:
    if i<=lastend or not valid[i-1]:continue
    end=None
    for j in range(i,min(i+4,n-1)+1):
     if not valid[j] or (ts[j]-ts[j-1])/np.timedelta64(1,'s')>1.5*step:break
     if j>i and abs(a[j]-a[i-1])<=.1:end=j;break
    short=end is not None
    if end is None:end=min(i+1,n-1)
    if not valid[i] or (ts[i]-ts[i-1])/np.timedelta64(1,'s')>1.5*step:continue
    if (ts[end]-ts[i])/np.timedelta64(1,'s')>7200:continue
    peak=i+int(np.argmax(abs(a[i:max(end,i+1)]-a[i-1])));amp=float(abs(a[peak]-a[i-1]));lastend=end if short else i
    l=max(0,i-2);h=min(n,end+3);sl=slice(l,h);goodh=(hu[sl]>0)&(hd[sl]>0)&np.isfinite(hu[sl])&np.isfinite(hd[sl]);goodq=np.isfinite(q[sl])&(q[sl]>=0);tr=target[l:h];target_ok=np.isfinite(tr).all() and (tr>=0).all();trange=rng(tr) if target_ok else np.nan
    huf=rng(hu[sl]) if goodh.all() else np.nan;hdf=rng(hd[sl]) if goodh.all() else np.nan;qrange=rng(q[sl]) if goodq.all() else np.nan;qmedian=float(np.median(q[sl])) if goodq.all() else np.nan;qth=max(5,.05*abs(qmedian)) if np.isfinite(qmedian) else np.nan
    quiet=bool(np.isfinite(huf+hdf+qrange+trange) and huf<=.05 and hdf<=.05 and qrange<=qth and trange<=.1 and abs(a[peak]-target[peak])>=1)
    other=[]
    for k in range(1,5):
     if k==hole:continue
     ar=g[f'孔{k}实际开度'].to_numpy()[sl];other.append(rng(ar))
    oth=max([v for v in other if np.isfinite(v)],default=np.nan)
    group='目标平稳且水位流量无明显响应的短脉冲' if quiet else '伴有响应或证据不足的短脉冲'
    ev=dict(station=name,code=code,hole=hole,start=str(g.时间.iloc[i]),peak_time=str(g.时间.iloc[peak]),end=str(g.时间.iloc[end]),source_row=int(g.源行号.iloc[peak]),baseline_m=float(a[i-1]),peak_m=float(a[peak]),amplitude_m=amp,target_at_peak=float(target[peak]),target_range_m=trange,hu_range_m=huf,hd_range_m=hdf,q_range_m3s=qrange,q_threshold_m3s=qth,other_holes_range_m=oth,cap_m=cap,event_type='2小时内回归的短脉冲' if short else '大幅阶跃或长脉冲边缘',target_all_zero=bool(np.nanmax(abs(target))==0) if np.isfinite(target).any() else None,classification=group.replace('短脉冲','开度突变'),window_start=str(g.时间.iloc[l]),window_end=str(g.时间.iloc[h-1]),in_author_30d=bool(pd.Timestamp('2025-07-22 12:00')<=g.时间.iloc[peak]<=pd.Timestamp('2025-08-21 12:00')))
    events.append(ev)
  # 各水位/流量孤立跳点：前后回到相近值。只作异常候选。
  for c,threshold,tol in [('闸前水位',.3,.05),('闸后水位',.3,.05),('瞬时流量',50,5)]:
   a=g[c].to_numpy();idx=np.flatnonzero((abs(a[1:-1]-a[:-2])>=threshold)&(abs(a[1:-1]-a[2:])>=threshold)&(abs(a[:-2]-a[2:])<=tol))+1
   for i in idx:
    if (g.时间.iloc[i+1]-g.时间.iloc[i-1]).total_seconds()>3*step:continue
    water_spikes.append(dict(station=name,code=code,field=c,time=str(g.时间.iloc[i]),source_row=int(g.源行号.iloc[i]),before=float(a[i-1]),value=float(a[i]),after=float(a[i+1])))
  if name in ['七里河倒虹吸出口节制闸','济河倒虹吸出口节制闸','金水河倒虹吸出口节制闸','黄金河倒虹吸出口节制闸','丈八沟倒虹吸出口节制闸']:
   gg=g[(g.时间>=pd.Timestamp('2025-07-22 12:00'))&(g.时间<=pd.Timestamp('2025-08-21 12:00'))].copy();gg['模型站号']=code;contexts.append(gg)
  print('gate',name,'events',len(events),flush=True)
 pd.DataFrame(events).to_csv(O/'开度突变逐事件.csv',index=False);pd.DataFrame(limits).to_csv(O/'闸孔台账与全年开度核对.csv',index=False);pd.DataFrame(water_spikes).to_csv(O/'水位流量孤立跳点.csv',index=False);pd.concat(contexts).to_csv(O/'五个重点站30天开度换算米.csv.gz',index=False,compression='gzip')
 (O/'节制闸审计进度.json').write_text(json.dumps(dict(raw_rows=nraw,events=len(events),suspect=sum(e['classification'].startswith('目标平稳') for e in events)),ensure_ascii=False,indent=2))

def others():
 for kind in ['分水口','惠南庄泵站']:
  df=pd.read_csv(O/(kind+'_全量原值.csv.gz'),parse_dates=['时间']);namecol='闸站名称';mass=[]
  for name,g,step,numeric in basic(df,kind,namecol):
   if kind=='分水口':
    parts=g[['孔1瞬时流量','孔2瞬时流量','孔3瞬时流量']];summed=parts.sum(axis=1,min_count=1);valid=(parts.fillna(0)>=0).all(axis=1)&(g.总瞬时流量>=0)&summed.notna();delta=abs(summed-g.总瞬时流量);mass.append(dict(station=name,valid_rows=int(valid.sum()),difference_gt_0p02=int((valid&(delta>.02)).sum()),max_difference=float(delta[valid].max()) if valid.any() else None))
  if mass:pd.DataFrame(mass).to_csv(O/'分水口总量与分孔一致性.csv',index=False)
  print(kind,'done',len(df),flush=True)

gates();others();pd.DataFrame(quality).to_csv(O/'全站全字段质量统计.csv',index=False);pd.DataFrame(coverage).to_csv(O/'时间覆盖与缺测统计.csv',index=False);pd.DataFrame(duplicates).to_csv(O/'重复时刻明细.csv',index=False)
print('DONE',len(quality),len(coverage),len(duplicates),flush=True)
