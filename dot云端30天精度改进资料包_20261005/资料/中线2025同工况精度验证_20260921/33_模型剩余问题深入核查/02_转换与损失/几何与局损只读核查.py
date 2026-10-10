"""复用冻结输入和现有构模入口，仅加载既有末态做代数核验，不进行水力推进。"""
from pathlib import Path
import hashlib, json, sys, subprocess
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
R=O.parents[1]
B=R/'26_冻结基线与后续归因'
B2=B/'B02_渠池归属修复后的参数同化诊断基线'
sys.path.insert(0,str(R/'30_分通道方法逐项测试/03_稳态与分通道试验'))
from 试验水力工具 import build, restore, STATE_KEYS
from pipedream_solver import ngeometry as ng

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
e,s,c,f,params=build(); m=e.model
links=c['links']; sl=c['superlinks'].set_index('id'); channels=s['baseData']['channels']
out={'scope':'只读代数/冻结状态几何重建；未运行时间步；未修改输入或基线', 'sources':{}}
project=R.parents[1]
for d in ['pipedream_solver','wxq_pipedream']:
 for frozen in (B/'共享依赖'/d).glob('*.py'):
  active=project/'water/pipedream-runner'/d/frozen.name
  out['sources'][str(active)]=dict(sha256=sha(active),matches_frozen=sha(active)==sha(frozen))
for name,active in [('physical_hook',R/'04_脚本/并联结构物理修复.py'),('runtime_common',R/'17_求解器修复/运行脚本/试验公共.py')]:
 frozen=B/('共享依赖/并联结构物理修复.py' if name=='physical_hook' else '共享依赖/水力运行脚本/试验公共.py')
 out['sources'][name]=dict(sha256=sha(active),matches_frozen=sha(active)==sha(frozen))

micro=[]
for shape,A,P,W,args in [('rect_open',ng.Rect_Open_A_ik,ng.Rect_Open_Pe_ik,ng.Rect_Open_B_ik,(9.1,20.)),('trapezoidal',ng.Trapezoidal_A_ik,ng.Trapezoidal_Pe_ik,ng.Trapezoidal_B_ik,(8.,20.,2.))]:
 for h in [args[0]-.5,args[0]+1.]:
  eps=1e-5
  micro.append(dict(shape=shape,h=h,g=args,area=float(A(h,h,*args)),perimeter=float(P(h,h,*args)),B=float(W(h,h,*args)),dA_dh_uniform=float((A(h+eps,h+eps,*args)-A(h-eps,h-eps,*args))/(2*eps)),length_m=100.,rise_m=.01,geometric_delta_V_m3=float(100*(A(h+.01,h+.01,*args)-A(h,h,*args))),continuity_delta_V_m3=float(100*W(h,h,*args)*.01)))
out['algebra_cases']=micro

profiles={}
for name,meta in c['channel_discretization'].items():
 if name not in channels:continue
 profiles.update({p['junction_id']:p for p in meta['sections']})
open_ids=links[links['shape'].isin(['rect_open','trapezoidal'])].index.to_numpy(int)
out['runtime']=dict(superlinks=int(m.NK),open_superlinks=len(channels),siphons=len(c['siphons_info']),links=len(links),open_links=len(open_ids),bc_method=m._bc_method,end_method=m._end_method,
 C_uk_unique=np.unique(m._C_uk).tolist(),C_dk_unique=np.unique(m._C_dk).tolist(),dx_uk_unique=np.unique(m._dx_uk).tolist(),dx_dk_unique=np.unique(m._dx_dk).tolist(),active_internal_controls=int(m._ctrl.sum()))
stats=[];clipped_rows=[];midpoint_rows=[];end_rows=[]
gx,gw=np.polynomial.legendre.leggauss(32);gx=(gx+1)/2;gw=gw/2
for label,path in [('B02六小时',B2/'六小时诊断起点/最新完整集合.npz'),('B02三十天末',B2/'同化运行/末态集合.npz')]:
 with np.load(path) as z:
  states={k:np.mean([z[f'm{i}_{k}'] for i in range(50)],axis=0) for k in STATE_KEYS}
  per_member=[]
  for member in range(50):
   h=z[f'm{member}_h_Ik'];ym=.5*(h[m._Ik]+h[m._Ip1k])
   per_member.append(int(np.count_nonzero(ym[open_ids]>m._g1_ik[open_ids]+1e-6)))
 restore(m,states)
 m.link_hydraulic_geometry();m.upstream_hydraulic_geometry();m.downstream_hydraulic_geometry()
 ym=.5*(m.h_Ik[m._Ik]+m.h_Ik[m._Ip1k]);ids=open_ids[ym[open_ids]>m._g1_ik[open_ids]+1e-6]
 geom_v=[];cont_v=[]
 for i in ids:
  row=links.loc[i]; h=ym[i]; g=(row.g1,row.g2) if row['shape']=='rect_open' else (row.g1,row.g2,row.g3)
  A=ng.Rect_Open_A_ik if row['shape']=='rect_open' else ng.Trapezoidal_A_ik
  dv=float((A(h+.01,h+.01,*g)-A(h,h,*g))*m._dx_ik[i]); bvol=float(m._B_ik[i]*m._dx_ik[i]*.01)
  geom_v.append(dv);cont_v.append(bvol)
  clipped_rows.append(dict(case=label,link_id=int(i),channel=sl.loc[row.k,'name'],shape=row['shape'],length_m=float(m._dx_ik[i]),mean_depth_m=float(h),g1_m=float(row.g1),above_g1_m=float(h-row.g1),A_m2=float(m._A_ik[i]),B_m=float(m._B_ik[i]),geometric_delta_V_for_uniform_1cm_m3=dv,continuity_delta_V_for_uniform_1cm_m3=bvol))
 stats.append(dict(case=label,checkpoint_sha256=sha(path),open_clipped_links=len(ids),channels=len(set(sl.loc[links.loc[ids].k,'name'])),per_member_count_min=min(per_member),per_member_count_max=max(per_member),geometry_dV_uniform_1cm_sum_m3=sum(geom_v),continuity_dV_uniform_1cm_sum_m3=sum(cont_v),clipped_length_m=float(m._dx_ik[ids].sum())))
 # 在原轮廓内，以沿程线性几何+线性水深的三次多项式精确积分作参照；不是现场真值。
 for i in open_ids:
  row=links.loc[i];pa=profiles[row.j_0];pb=profiles[row.j_1];ha=float(m.h_Ik[row.j_0]);hb=float(m.h_Ik[row.j_1])
  if min(ha,hb)<=0 or ha>pa['height'] or hb>pb['height']:continue
  def area(t):
   h=(1-t)*ha+t*hb; b=(1-t)*pa['bottom_width']+t*pb['bottom_width']; ss=(1-t)*pa['side_slope']+t*pb['side_slope']; return h*(b+ss*h)
  simpson=(area(0)+4*area(.5)+area(1))/6
  midpoint=float(m._A_ik[i]); delta=midpoint-simpson
  hh=(1-gx)*ha+gx*hb;bb=(1-gx)*pa['bottom_width']+gx*pb['bottom_width'];ss=(1-gx)*pa['side_slope']+gx*pb['side_slope']
  aa=hh*(bb+ss*hh);rr=aa/(bb+2*hh*np.sqrt(1+ss**2))
  resistance_integral=float(gw@(1/(aa**2*rr**(4/3))))
  resistance_midpoint=float(1/(m._A_ik[i]**2*m._R_ik[i]**(4/3)))
  midpoint_rows.append(dict(case=label,link_id=int(i),channel=sl.loc[row.k,'name'],length_m=float(m._dx_ik[i]),area_midpoint_m2=midpoint,area_longitudinal_integral_m2=simpson,delta_area_m2=delta,relative_area_error=delta/simpson,relative_friction_error=resistance_midpoint/resistance_integral-1,delta_volume_m3=delta*m._dx_ik[i],left_depth_m=ha,right_depth_m=hb,left_width_m=pa['bottom_width'],right_width_m=pb['bottom_width'],left_slope=pa['side_slope'],right_slope=pb['side_slope']))
 for k,row in sl.iterrows():
  hu=float(m._C_uk[k]*m.Q_uk[k]*abs(m.Q_uk[k])/(2*9.81*m._A_uk[k]**2))
  hd=float(m._C_dk[k]*m.Q_dk[k]*abs(m.Q_dk[k])/(2*9.81*m._A_dk[k]**2))
  info=next((a for a in c['siphons_info'] if a['superlink_id']==k),None)
  ids_k=np.flatnonzero(m._ki==k)
  explicit=0. if info is None else float(np.sum(info['loss_coeff']*m._dx_ik[ids_k]/m._dx_ik[ids_k].sum()*m.Q_ik[ids_k]*abs(m.Q_ik[ids_k])/(2*9.81*m._A_ik[ids_k]**2)))
  end_rows.append(dict(case=label,superlink_id=int(k),channel=row['name'],kind='siphon' if info else 'open_channel',upstream_default_head_term_m=hu,downstream_default_head_term_m=hd,default_terms_sum_m=hu+hd,explicit_source_K=None if info is None else info['loss_coeff'],explicit_source_K_head_term_m=explicit,upstream_area_m2=float(m._A_uk[k]),downstream_area_m2=float(m._A_dk[k])))
out['clipping']=stats
out['midpoint_summary']=[]
for label in [x['case'] for x in stats]:
 a=[x for x in midpoint_rows if x['case']==label]
 out['midpoint_summary'].append(dict(case=label,within_raw_contour_links=len(a),max_abs_relative_area_error=max(abs(x['relative_area_error']) for x in a),max_abs_relative_friction_error=max(abs(x['relative_friction_error']) for x in a),sum_delta_volume_m3=sum(x['delta_volume_m3'] for x in a),sum_abs_delta_volume_m3=sum(abs(x['delta_volume_m3']) for x in a),examples=sorted(a,key=lambda x:abs(x['relative_area_error']),reverse=True)[:5],friction_examples=sorted(a,key=lambda x:abs(x['relative_friction_error']),reverse=True)[:5]))
out['end_term_summary']=[]
for label in [x['case'] for x in stats]:
 for kind in ['open_channel','siphon']:
  a=[x for x in end_rows if x['case']==label and x['kind']==kind]
  out['end_term_summary'].append(dict(case=label,kind=kind,count=len(a),sum_abs_default_terms_m=sum(abs(x['upstream_default_head_term_m'])+abs(x['downstream_default_head_term_m']) for x in a),max_abs_default_pair_m=max(abs(x['default_terms_sum_m']) for x in a),sum_abs_explicit_source_K_terms_m=sum(abs(x['explicit_source_K_head_term_m']) for x in a),top_examples=sorted(a,key=lambda x:abs(x['default_terms_sum_m']),reverse=True)[:5],sum_warning='按各段局部名义方向计算，汇总只展示项量级，不能当全线可加的同一方向实测总水头差'))
doc_python='/Users/gaobingxiang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
tab=json.loads(subprocess.check_output([doc_python,'-c',"import openpyxl,json,sys; w=openpyxl.load_workbook(sys.argv[1],read_only=True,data_only=True); print(json.dumps(list(w['渐变段'].values),ensure_ascii=False)); w.close()",str(R/'01_原始资料/中线全线建模数据.xlsx')],text=True))
out['transition_source_headers']=[str(x) for x in tab[0]]
out['transition_source_count']=len(tab)-1
out['transition_source_example_rows']=[list(x) for x in tab[1:6]]
out['transition_coefficients']={chr(65+i)+'列_'+str(tab[0][i]):dict(nonzero_count=sum(isinstance(x[i],(float,int)) and x[i]!=0 for x in tab[1:]),unique_values=sorted(set(x[i] for x in tab[1:] if isinstance(x[i],(float,int))))) for i in [10,11]}
out['source_channel_keys']=sorted(set().union(*(set(v) for v in channels.values())))
out['source_section_keys']=sorted(set().union(*(set(v) for v in s['baseData']['sections'].values())))
for name,rows in [('截高连接明细',clipped_rows),('沿程中点近似明细',midpoint_rows),('各段端部水头项明细',end_rows)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
(O/'几何与局损只读核查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({k:out[k] for k in ['runtime','clipping','midpoint_summary','end_term_summary','transition_source_headers','transition_source_count','transition_coefficients']},ensure_ascii=False,indent=2))
