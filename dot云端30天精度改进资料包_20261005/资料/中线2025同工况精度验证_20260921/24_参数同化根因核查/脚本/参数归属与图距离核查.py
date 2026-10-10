from pathlib import Path
import sys,json,numpy as np,pandas as pd
R=Path.cwd()/'outputs/中线2025同工况精度验证_20260921';P=R/'23_五十成员参数同化';O=R/'24_参数同化根因核查'
sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(P/'脚本')]
from 试验公共 import meshes
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import ParameterEnKF
e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()))
ens=ParameterEnKF(e,c);m=e.model;mesh=meshes(s,c);rows=[]
for name,mm in mesh.items():
 k=mm['k'];row=c['superlinks'].iloc[k];ids=np.flatnonzero(m._ki==k)
 rows.append(dict(name=name,k=int(k),row_id=int(row.id),actual_start=mm['x'].min(),actual_end=mm['x'].max(),graph_start=ens.xparts['H_j'][m._J_uk[k]],graph_end=ens.xparts['H_j'][m._J_dk[k]],internal_x_min=ens.xparts['h_Ik'][mm['ids']].min(),internal_x_max=ens.xparts['h_Ik'][mm['ids']].max(),link_ids_min=int(ids.min()),link_ids_max=int(ids.max())))
pd.DataFrame(rows).to_csv(O/'渠段桩号与图距离.csv',index=False);c['superlinks'].to_csv(O/'超级渠段转换表.csv',index=False)
print('pool3',pd.DataFrame(rows)[pd.DataFrame(rows).name.str.match(r'POOL3(?:\D|$)')].to_string(index=False))
ops=json.loads((P/'P01M4_50成员30天参数同化/观测算子与留出.json').read_text());oprows=[]
for op in ops:
 x=float(ens.xparts[op['state']][op['indices']]@op['weights']);oprows.append(dict(station=op['station'],metric=op['metric'],actual=op['chainage'],graph=x,assimilated=op['assimilated']))
pd.DataFrame(oprows).to_csv(O/'同化测点图距离.csv',index=False)
# Filter grouping must match individual hydraulic link ownership.
for par in ens.parameters:
 if par['kind']=='n':
  actualks=np.unique(m._ki[par['indices']]);print(par['name'],len(par['indices']),actualks.tolist(),round(par['xmin']/1000,2),round(par['xmax']/1000,2))
print('columns',c['superlinks'].columns.tolist())
side=pd.read_csv(P/'30天参数同化逐侧精度.csv');f=side[side.run.str.startswith('P01')];b=side[side.run.str.startswith('P02')];merged=f.merge(b,on=['operator','station','metric'],suffixes=('_a','_b'));merged['delta_mae']=merged.mae_a-merged.mae_b;merged.sort_values('delta_mae',ascending=False).to_csv(O/'逐测点改善恶化.csv',index=False)
print(merged.sort_values('delta_mae',ascending=False)[['operator','station','metric','mae_a','mae_b','delta_mae']].head(15).to_string(index=False))
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
rr=np.r_[m._J_uk,m._J_dk];cc=np.r_[m._J_dk,m._J_uk];count,lab=connected_components(csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(m.M,m.M)),directed=False)
components=sorted(set(lab[m._J_uk]),key=lambda z:ens.xparts['H_j'][lab==z].min())
print('components',count,len(components))
assignment={int(k):components.index(lab[m._J_uk[k]])+1 for k in range(len(m._J_uk))}
print('row23 topology pool',assignment[23]);changes=[]
for par in ens.parameters:
 if par['kind']!='n':continue
 for k in np.unique(m._ki[par['indices']]):
  if assignment[int(k)]!=int(par['name'][4:-2]):changes.append(dict(name=c['superlinks'].iloc[k]['name'],k=int(k),old_pool=int(par['name'][4:-2]),new_pool=assignment[int(k)]))
print('changes',changes);(O/'拓扑归属核验.json').write_text(json.dumps(dict(components=len(components),changes=changes),ensure_ascii=False,indent=2))
