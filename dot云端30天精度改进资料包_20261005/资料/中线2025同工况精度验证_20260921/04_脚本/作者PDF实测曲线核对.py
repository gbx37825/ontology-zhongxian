"""Calibrate vector curves from PDF axis ticks, never by regressing observations."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd,pdfplumber
R=Path(__file__).resolve().parents[1];raw=pd.read_csv(R/'02_输入/实测评分长表.csv');raw=raw[raw.metric.isin(['forebay_m','afterbay_m'])];stats=[];points=[];audit=[]
def decode(t):
 if t.startswith('/uni'):code=int(t[4:],16)
 elif t.startswith('\x00') and len(t)==2:code=ord(t[1])
 else:return t
 return str(code-0x16) if 0x16<=code<=0x1f else '.' if code==0x14 else '-' if code==0x13 else ''
with pdfplumber.open(R/'01_原始资料/stations_30d_flow.pdf') as pdf:
 for pi,p in enumerate(pdf.pages):
  sid=f'JZZ{pi+2}';panels=[r for r in p.rects if r['width']>500 and r.get('non_stroking_color')==1. and r['top']>1 and r['top']<100]
  if len(panels)!=1:audit.append({'station':sid,'status':'上面板不唯一'});continue
  panel=panels[0];x0=panel['x0'];top=panel['top'];bottom=panel['bottom'];grids=sorted(set(round(l['top'],6) for l in p.lines if l['width']>500 and l['height']<1e-5 and top<l['top']<bottom and l.get('stroking_color')==.6901960784))
  groups={}
  for c in p.chars:
   v=decode(c['text'])
   if v and c['x1']<x0-3 and c['x0']>x0-45 and top-8<c['top']<bottom:groups.setdefault(round(c['top'],3),[]).append((c['x0'],v))
  ticks=[]
  for y in grids:
   key=min(groups,key=lambda t:abs(t+3.44-y));txt=''.join(v for x,v in sorted(groups[key]))
   if abs(key+3.44-y)>1 or not re.fullmatch(r'-?\d+(\.\d+)?',txt):continue
   ticks.append((y,float(txt)))
  if len(ticks)<2:audit.append({'station':sid,'status':'纵轴刻度不足','groups':groups});continue
  coef=np.polyfit(np.array(ticks)[:,0],np.array(ticks)[:,1],1);err=np.max(abs(np.polyval(coef,np.array(ticks)[:,0])-np.array(ticks)[:,1]));assert err<1e-5,(sid,ticks,err)
  xgrid=sorted(set(l['x0'] for l in p.lines if l['height']>150 and l['top']==top and abs(l['bottom']-bottom)<1e-4 and l.get('stroking_color')==.6901960784));assert len(xgrid)==7,(sid,xgrid);assert np.ptp(np.diff(xgrid))<1e-4
  audit.append({'station':sid,'page':pi+1,'status':'独立轴校准通过','y_ticks':ticks,'y_to_Z':coef.tolist(),'max_axis_fit_error_m':err,'x_grid':xgrid,'x_tick_days':[0,5,10,15,20,25,30]})
  for met,color in [('forebay_m',.5019607843),('afterbay_m',.4117647059)]:
   paths=[q for q in p.curves if q['width']>500 and q['top']>=top-1e-3 and q['bottom']<=bottom+1e-3 and isinstance(q.get('stroking_color'),(int,float)) and abs(q['stroking_color']-color)<1e-8]
   if len(paths)!=1:continue
   xy=np.asarray(paths[0]['pts']);t=(xy[:,0]-xgrid[0])/(xgrid[-1]-xgrid[0])*30;z=np.polyval(coef,xy[:,1]);assert np.all(np.diff(t)>=0)
   o=raw[(raw.station_code==sid)&(raw.metric==met)].sort_values('time_s');tt=o.time_s.to_numpy()/86400;v=o.value.to_numpy();estimated=np.interp(tt,t,z);valid=np.isfinite(v)&(v>-100)&(tt>=t[0]-1e-7)&(tt<=t[-1]+1e-7);dif=estimated[valid]-v[valid]
   median=np.median(dif);centered=dif-median
   lags=[]
   for lag in range(-48,49):
    shifted=tt+lag/48;ok=valid&(shifted>=0)&(shifted<=30);dd=np.interp(shifted[ok],t,z)-v[ok];lags.append((float(np.mean(abs(dd))),lag))
   best=min(lags);row=dict(station_code=sid,page=pi+1,metric=met,n=int(valid.sum()),vector_vertices=len(t),PDF_minus_raw_mean=float(dif.mean()),PDF_minus_raw_median=float(median),MAE=float(abs(dif).mean()),P95_abs=float(np.quantile(abs(dif),.95)),max_abs=float(abs(dif).max()),centered_P95=float(np.quantile(abs(centered),.95)),best_shift_hours=best[1]/2,best_shift_MAE=best[0],offset_02_035_candidate=bool(.2<=median<=.35 and np.quantile(abs(centered),.95)<.02));aligned_ok=valid; aligned=np.interp(tt,t,z); ad=aligned[aligned_ok]-v[aligned_ok]; am=float(np.median(ad)); row.update(aligned_n=int(aligned_ok.sum()),aligned_MAE=float(np.mean(abs(ad))),aligned_median_bias=am,aligned_mean_bias=float(ad.mean()),aligned_P95_abs=float(np.quantile(abs(ad),.95)),aligned_centered_P95=float(np.quantile(abs(ad-am),.95)),aligned_max_abs=float(abs(ad).max()),offset_02_035_candidate=bool(.2<=am<=.35 and np.quantile(abs(ad-am),.95)<.02));stats.append(row)
   for ti,vi,ei,ok,ai,aok in zip(tt,v,estimated,valid,aligned,aligned_ok):points.append(dict(station_code=sid,metric=met,day=ti,raw=vi,pdf_estimate=ei,difference=ei-vi,valid=bool(ok),aligned_pdf_estimate=ai if aok else None,aligned_difference=ai-vi if aok else None,aligned_valid=bool(aok)))
   if sid=='JZZ49' and met=='afterbay_m':audit.append({'station':sid,'anchor':'闸后孤立低尖峰','PDF_day_at_min':float(t[np.argmin(z)]),'PDF_Zmin':float(z.min()),'raw_day_at_min':float(tt[np.argmin(v)]),'raw_day_from_0722_midnight':float(tt[np.argmin(v)]+.5),'raw_time_origin':'2025-07-22 12:00','raw_Zmin':float(v.min()),'visual_checked_page':48,'render':'渲染确认上图灰色闸后线约23.94天从约75m跌至约70m'})
f=pd.DataFrame(stats);f.to_csv(R/'07_对比分析/作者PDF与2025实测一致性逐侧.csv',index=False);pd.DataFrame(points).to_csv(R/'07_对比分析/作者PDF与2025实测一致性逐点.csv',index=False)
j={'axis_source':'PDF纵轴灰网格线位置+左侧刻度字符独立拟合；无原数据坐标回归。横轴为渲染确认的0/5/10/15/20/25/30天网格，起点2025-07-22 12:00。','curve_colors':{'forebay':.5019607843,'afterbay':.4117647059},'series_count':len(f),'mean_series_MAE':float(f.MAE.mean()),'aligned_mean_series_MAE':float(f.aligned_MAE.mean()),'alignment':'尖峰独立绑定：旧00时窗口通过JZZ49尖峰独立确认PDF原点为07-22 12:00；本版原始窗口已统一07-22 12:00至08-21 12:00，PDF_day=当前raw_day，无需额外时移，完整30天。','max_abs_median_bias':float(f.PDF_minus_raw_median.abs().max()),'offset_02_035_candidates':f[f.offset_02_035_candidate].to_dict('records'),'audit':audit,'missing_gray_sides':[{'station_code':f'JZZ{n}','metric':met} for n in range(2,61) for met in ['forebay_m','afterbay_m'] if not ((f.station_code==f'JZZ{n}')&(f.metric==met)).any()],'visual_checked_pages':[8,47,48],'limitation':'图线简化造成局部插值误差；结果为图上估计，不作为原始实测修正。时移搜索仅±24h、30min步长，平坦曲线最佳时移可能不唯一。'};(R/'07_对比分析/作者PDF与2025实测一致性审核.json').write_text(json.dumps(j,ensure_ascii=False,indent=2))
ls=['# 作者PDF与2025实测曲线一致性','',f'成功独立标定并比较{len(f)}条灰色水位曲线；其他侧未检出可比灰色路径，不用模型曲线冒充实测。旧00时窗口尖峰独立锁定作者起报为12h。本版统一12h起止后，共同比较完整30天，逐侧MAE均值{f.aligned_MAE.mean():.8f}m。固定上移0.2—0.35m且中心残差P95<0.02m的侧数为{int(f.offset_02_035_candidate.sum())}。','',j['axis_source'],'',j['alignment'],'',j['limitation'],'','## 尖峰横轴绑定','',json.dumps([a for a in audit if 'anchor'in a],ensure_ascii=False),'','## 固定高程偏移','', 'JZZ9黄金河闸前、闸后均为PDF实测相对原始实测+0.245m；JZZ48古运河闸后为+0.301m。这里确认的是2站3侧，不应改称3站。其余可提取灰线的侧别未发现同量级固定偏移；缺灰线侧别不作判断。第8、47、48页已渲染核验数字刻度、灰线身份和孤立尖峰。','', '## 偏差较大的侧别','','```text\n'+f.sort_values('aligned_MAE',ascending=False).head(15).to_string(index=False)+'\n```']
(R/'08_报告/作者PDF与2025实测一致性审核.md').write_text('\n'.join(ls));print(f.sort_values('aligned_MAE',ascending=False).head(15).to_string(index=False));print('SUMMARY',j['series_count'],j['mean_series_MAE'],j['max_abs_median_bias'],j['offset_02_035_candidates']);print([a for a in audit if 'anchor'in a])
