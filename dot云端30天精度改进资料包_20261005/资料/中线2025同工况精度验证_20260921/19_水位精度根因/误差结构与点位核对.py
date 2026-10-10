from pathlib import Path
import sys,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'19_水位精度根因'
sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import build_engine,meshes,weights_at
obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');run=R/'06_结果/S02_C07输入及闸流修复严格30天';raw=pd.read_csv(run/'逐闸水位流量.csv');d=raw.melt(id_vars=['station_code','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs,on=['station_code','metric','time_s'],validate='one_to_one');d=d[d.valid];d['err']=d.sim-d.value;rows=[]
for (st,side),g in d.groupby(['station_code','metric']):
 v=g[g.time_s>0];bias=v.err.mean();rows.append(dict(station=st,side=side,pdf_comparable=bool(v.pdf_comparable.iloc[0]),mae=float(abs(v.err).mean()),bias=float(bias),centered_mae_diagnostic=float(abs(v.err-bias).mean()),initial_error=float(g[g.time_s==0].err.iloc[0]),mean_first3d=float(v[v.time_s<=259200].err.mean()),mean_last3d=float(v[v.time_s>2332800].err.mean())))
stats=pd.DataFrame(rows);stats.to_csv(O/'S02逐侧误差结构.csv',index=False)
x=d[d.time_s>0].pivot(index=['station_code','time_s'],columns='metric',values='err');x['common_bias']=(x.forebay_m+x.afterbay_m)/2;x['drop_error']=x.forebay_m-x.afterbay_m;x.groupby('station_code').agg(common_bias=('common_bias','mean'),mean_abs_drop_error=('drop_error',lambda z:abs(z).mean()),drop_bias=('drop_error','mean')).to_csv(O/'同闸共同偏差与水位差误差.csv')
summary=dict(mae88=float(stats[stats.pdf_comparable].mae.mean()),bias88=float(stats[stats.pdf_comparable].bias.mean()),negative_bias_sides=int((stats.bias<0).sum()),sides=len(stats),centered_mae88=float(stats[stats.pdf_comparable].centered_mae_diagnostic.mean()),initial_error_vs_mean_bias_correlation=float(stats.initial_error.corr(stats.bias)),common_bias_mean_abs=float(abs(x.common_bias).mean()),gate_drop_error_mae=float(abs(x.drop_error).mean()),note='去均值MAE只用于诊断偏差形态，不作为改正后精度，未改任何观测')
# 使用原安装表已审定的同侧唯一位置候选，核验本次模型网格能否承接。
mapfile=R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/水位测点位置审核.json';mapping=json.loads(mapfile.read_text());e,s,c=build_engine(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));mesh=meshes(s,c);sn=np.load(run/'内部断面状态.npz');H=sn['h_Ik']+e.model._z_inv_Ik;times=sn['time_s'];pos=[];failed=[]
for a in mapping['station_side_summary']:
 if not a['candidate_interpolation']:continue
 rs=[q for q in mapping['sensor_rows'] if q['stationcode']==a['stationcode'] and q['metric']==a['metric'] and q['status']=='可作位置插值候选'];r=rs[0];ch=r['model_brackets'][0]['channel'];x0=r['sensor_chainage'];met='forebay_m' if a['metric']=='front' else 'afterbay_m'
 if ch not in mesh:failed.append([a['stationcode'],met,'网格缺渠道']);continue
 ids,w=weights_at(mesh[ch],x0);z=H[:,ids]@w;assert abs(w.sum()-1)<1e-12
 for t,h in zip(times,z):pos.append(dict(station_code=a['stationcode'],metric=met,time_s=t,position_level=float(h),chainage=x0))
pd0=pd.DataFrame(pos).merge(d,on=['station_code','metric','time_s'],validate='one_to_one');pd0=pd0[pd0.time_s>0];pd0['position_error']=pd0.position_level-pd0.value
pd0.to_csv(O/'2025同侧测点位置对照.csv',index=False);summary['position_candidate_sides']=pd0.groupby(['station_code','metric']).ngroups;summary['same_subset_gate_mae']=float(abs(pd0.err).mean());summary['same_subset_sensor_mae']=float(abs(pd0.position_error).mean());summary['max_position_change']=float(abs(pd0.position_level-pd0.sim).max());summary['mean_position_change']=float(abs(pd0.position_level-pd0.sim).mean());summary['position_failures']=failed
(O/'误差结构与点位摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2));print(stats.nlargest(10,'mae').to_string(index=False))
