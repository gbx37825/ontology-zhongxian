"""Reuse independent axis calibration, distinguish deterministic and EnKF PDF estimates."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pdfplumber
R=Path(__file__).resolve().parents[1];grid=np.arange(1441)*1800.;audit=json.loads((R/'07_对比分析/作者PDF与2025实测一致性审核.json').read_text());axes={q['station']:q for q in audit['audit'] if q.get('status')=='独立轴校准通过'};colors={'forebay_m':(.1215686275,.4666666667,.7058823529),'afterbay_m':(.8392156863,.1529411765,.1568627451)};rows=[];errs=[];coverage=[]
def sample(path,x0,x1,convert):
 pts=np.asarray(path['pts']);t=(pts[:,0]-x0)/(x1-x0)*2592000;v=convert(pts[:,1]);assert np.all(np.diff(t)>=0)
 if abs(t[0])<.01:t[0]=0
 if abs(t[-1]-2592000)<.01:t[-1]=2592000
 ok=(grid>=t[0])&(grid<=t[-1]);e=np.full(1441,np.nan);e[ok]=np.interp(grid[ok],t,v);return e,ok
with pdfplumber.open(R/'01_原始资料/stations_30d_flow.pdf') as pdf:
 for i,p in enumerate(pdf.pages):
  sid=f'JZZ{i+2}';ax=axes[sid];panels=[q for q in p.rects if q['width']>500 and q.get('non_stroking_color')==1. and q['top']>1];panels=sorted(panels,key=lambda x:x['top']);assert len(panels)==2
  bands=[q for q in p.rects if q['width']>500 and isinstance(q.get('non_stroking_color'),tuple) and np.allclose(q['non_stroking_color'],(0,.5019607843,0))];assert len(bands)==1;b=bands[0];center=(b['top']+b['bottom'])/2;scale=(b['bottom']-b['top'])/.2
  for metric,color in colors.items():
   for panel,kind in zip(panels,['water_level','error']):
    paths=[q for q in p.curves if q['width']>500 and q['top']>=panel['top']-1e-3 and q['bottom']<=panel['bottom']+1e-3 and isinstance(q.get('stroking_color'),tuple) and np.allclose(q['stroking_color'],color)]
    coverage.append(dict(station_code=sid,metric=metric,kind=kind,path_count=len(paths)))
    if not paths:continue
    assert len(paths)==1,(sid,metric,kind,len(paths));vals,ok=sample(paths[0],ax['x_grid'][0],ax['x_grid'][-1],lambda y:np.polyval(ax['y_to_Z'],y) if kind=='water_level' else (center-y)/scale)
    dest=rows if kind=='water_level' else errs
    for t,v,cov in zip(grid,vals,ok):dest.append(dict(station_code=sid,metric=metric,time_s=int(t),value_estimate=v,covered=bool(cov),source_page=i+1))
h=pd.DataFrame(rows);er=pd.DataFrame(errs);h.rename(columns={'value_estimate':'water_level_estimate'}).to_csv(R/'07_对比分析/作者PDF确定性水位同网格估计.csv',index=False);er.rename(columns={'value_estimate':'error_estimate'}).to_csv(R/'07_对比分析/作者PDF确定性误差同网格估计.csv',index=False)
obs=pd.read_csv(R/'02_输入/实测评分长表.csv');obs=obs[obs.time_s==0][['station_code','metric','value']].rename(columns={'value':'raw_observation_H0'});initial=h[h.time_s==0].rename(columns={'value_estimate':'deterministic_H0'}).merge(obs,on=['station_code','metric'],how='left',validate='one_to_one');offset=json.loads((R/'07_对比分析/作者已核实水位偏移.json').read_text())['offsets'];off={(x['station_code'],x['metric']):x['offset_m'] for x in offset};initial['author_offset']=initial.apply(lambda r:off.get((r.station_code,r.metric),0.),axis=1);initial['author_observation_H0']=initial.raw_observation_H0+initial.author_offset;initial['H0_minus_raw']=initial.deterministic_H0-initial.raw_observation_H0;initial['H0_minus_author_adjusted']=initial.deterministic_H0-initial.author_observation_H0;initial.to_csv(R/'07_对比分析/作者PDF确定性初始水位估计.csv',index=False)
s=er.groupby(['station_code','metric']).value_estimate.agg(n='count',MAE=lambda x:abs(x).mean(),max_abs=lambda x:abs(x).max()).reset_index();enkf=pd.read_csv(R/'07_对比分析/作者PDF同网格逐侧误差估计.csv');comp=s.merge(enkf[['station_code','metric','MAE_estimate']],on=['station_code','metric'],validate='one_to_one').rename(columns={'MAE':'deterministic_MAE','MAE_estimate':'EnKF_MAE'});comp['EnKF_minus_deterministic']=comp.EnKF_MAE-comp.deterministic_MAE;comp.to_csv(R/'07_对比分析/作者PDF确定性与EnKF同侧对比.csv',index=False)
j={'water_level_sides':h.groupby(['station_code','metric']).ngroups,'error_sides':er.groupby(['station_code','metric']).ngroups,'common_sides':len(comp),'deterministic_MAE_mean':float(comp.deterministic_MAE.mean()),'EnKF_MAE_mean':float(comp.EnKF_MAE.mean()),'EnKF_improved_sides':int((comp.EnKF_minus_deterministic<0).sum()),'H0_common_n':len(initial),'H0_author_adjusted_MAE':float(initial.H0_minus_author_adjusted.abs().mean()),'H0_author_adjusted_max_abs':float(initial.H0_minus_author_adjusted.abs().max()),'coverage':coverage,'interpretation':'确定性与EnKF误差均为图上相同88侧、1441点估计；水位118侧仅在实际有路径时输出。初始误差接近零也只能说明图示初始吻合，不能直接证明作者初始化实现。图例/缺曲线不填零；不外推。'};(R/'07_对比分析/作者PDF确定性提取审核.json').write_text(json.dumps(j,ensure_ascii=False,indent=2));print({k:v for k,v in j.items() if k!='coverage'})
