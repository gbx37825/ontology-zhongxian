"""Sample author PDF vector error curves; estimates, not original solver output."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pdfplumber
R=Path(__file__).resolve().parents[1];grid=np.arange(1441,dtype=float)*1800;rows=[];scores=[]
colors={'forebay_m':(.0901960784,.7450980392,.8117647059),'afterbay_m':(1.,.4980392157,.0549019608)}
with pdfplumber.open(R/'01_原始资料/stations_30d_flow.pdf') as pdf:
 for i,p in enumerate(pdf.pages):
  bands=[q for q in p.rects if q['width']>500 and isinstance(q['non_stroking_color'],tuple) and np.allclose(q['non_stroking_color'],(0,.5019607843,0))]
  assert len(bands)==1
  b=bands[0];center=(b['top']+b['bottom'])/2;scale=(b['bottom']-b['top'])/.2
  panels=[q for q in p.rects if q['width']>500 and q['top']>250 and q.get('non_stroking_color')==1.];assert len(panels)==1;panel=panels[0]
  xgrid=sorted(set(l['x0'] for l in p.lines if abs(l['top']-panel['top'])<1e-4 and abs(l['bottom']-panel['bottom'])<1e-4 and l['height']>50 and l.get('stroking_color')==.6901960784));assert len(xgrid)==7
  for metric,color in colors.items():
   paths=[q for q in p.curves if q['width']>500 and q['top']>=panel['top']-1e-3 and q['bottom']<=panel['bottom']+1e-3 and isinstance(q.get('stroking_color'),tuple) and np.allclose(q['stroking_color'],color)]
   if not paths:continue
   assert len(paths)==1,(i,metric,len(paths));xy=np.asarray(paths[0]['pts']);t=(xy[:,0]-xgrid[0])/(xgrid[-1]-xgrid[0])*2592000;err=(center-xy[:,1])/scale;assert np.all(np.diff(t)>=0)
   # Coordinate rounding at declared tick endpoints is tolerance only, never extending a real short path.
   if abs(t[0])<.01:t[0]=0
   if abs(t[-1]-2592000)<.01:t[-1]=2592000
   ok=(grid>=t[0])&(grid<=t[-1]);estimate=np.full(len(grid),np.nan);estimate[ok]=np.interp(grid[ok],t,err)
   for tt,ee,valid in zip(grid,estimate,ok):rows.append(dict(station_code=f'JZZ{i+2}',metric=metric,time_s=int(tt),error_estimate=ee,covered=bool(valid),source_page=i+1))
   scores.append(dict(station_code=f'JZZ{i+2}',metric=metric,covered_n=int(ok.sum()),MAE_estimate=float(np.nanmean(abs(estimate))),max_abs_estimate=float(np.nanmax(abs(estimate))),vector_max_abs=float(abs(err).max()),vector_max_time_s=float(t[np.argmax(abs(err))]),path_vertices=len(t),first_time_s=float(t[0]),last_time_s=float(t[-1])))
f=pd.DataFrame(rows);s=pd.DataFrame(scores);assert len(s)==88,len(s);assert len(f)==88*1441
f.to_csv(R/'07_对比分析/作者PDF同网格误差估计.csv',index=False);s.to_csv(R/'07_对比分析/作者PDF同网格逐侧误差估计.csv',index=False)
a={'series':len(s),'grid_points_per_series':1441,'time_start':'2025-07-22 12:00:00','time_end':'2025-08-21 12:00:00','mean_series_MAE_estimate':float(s.MAE_estimate.mean()),'uncovered_points':int((~f.covered).sum()),'source':'PDF下误差面板青色闸前/橙色闸后；绿色带上下界±0.1m独立纵轴标定，横轴0..30天网格','limitation':'矢量曲线简化后的线性插值近似，不是作者原始输出；不把图例或缺线当0，不外推未覆盖端点','JZZ49_afterbay':s[(s.station_code=='JZZ49')&(s.metric=='afterbay_m')].to_dict('records')};assert abs(a['JZZ49_afterbay'][0]['vector_max_time_s']-2068200)<.01;(R/'07_对比分析/作者PDF同网格误差估计审核.json').write_text(json.dumps(a,ensure_ascii=False,indent=2));print(a)
