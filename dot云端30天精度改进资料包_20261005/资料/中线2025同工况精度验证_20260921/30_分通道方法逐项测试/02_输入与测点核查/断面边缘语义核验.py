"""只读核查输入断面边缘筛查语义；不改变断面，不重新计算水力过程。"""
from pathlib import Path
import sys,json
import numpy as np
O=Path(__file__).resolve().parent
D=O.parent
sys.path.insert(0,str(D/'03_稳态与分通道试验'))
from 试验水力工具 import build,restore,STATE_KEYS
from wxq_pipedream.section_geometry import classify_section,interpolate_section_geometry,to_pipedream_geometry
e,s,c,f,params=build();m=e.model
v=json.loads((D/'03_稳态与分通道试验/目标稳态超断面边缘明细.json').read_text())
with np.load(D/'03_稳态与分通道试验/目标时刻恒边界稳态_状态.npz') as z:restore(m,{k:z[k].copy() for k in STATE_KEYS})
m.link_hydraulic_geometry()
records=[]
for row in v:
 ch=row['channel'];p=next(a for a in c['channel_discretization'][ch]['sections'] if a['junction_id']==row['junction']);a,b=p['source_interval'];sa=s['baseData']['sections'][a];sb=s['baseData']['sections'][b]
 g=interpolate_section_geometry(classify_section(sa['yz']),classify_section(sb['yz']),p['fraction']);j=row['junction'];edge=(1-p['fraction'])*min(sa['yz'][0][1],sa['yz'][-1][1])+p['fraction']*min(sb['yz'][0][1],sb['yz'][-1][1])
 records.append(dict(**row,source_interval=[a,b],fraction=p['fraction'],shape=g.shape_kind,both_raw_endpoints_equal=bool(sa['yz'][0][1]==sa['yz'][-1][1] and sb['yz'][0][1]==sb['yz'][-1][1]),source_edge_m=edge,converted_section_top_m=g.z_inv+g.height,source_vs_converted_top_diff_m=g.z_inv+g.height-edge,computed_stage_m=float(m.h_Ik[j]+m._z_inv_Ik[j])))
worst_channels=list(dict.fromkeys(x['channel'] for x in sorted(v,key=lambda x:x['above_model_section_edge_m'],reverse=True)))[:3]
samples=[]
for ch in worst_channels:
 row=max([x for x in records if x['channel']==ch],key=lambda x:x['above_model_section_edge_m']);a,b=row['source_interval'];adj=[]
 for link in c['links'][(c['links'].j_0==row['junction'])|(c['links'].j_1==row['junction'])].itertuples():
  hmean=float((m.h_Ik[link.j_0]+m.h_Ik[link.j_1])/2);hc=min(max(hmean,0),link.g1)
  expected=hc*(link.g2+link.g3*hc) if link.shape=='trapezoidal' else hc*link.g2
  adj.append(dict(id=link.id,shape=link.shape,mean_depth_m=hmean,g1_m=link.g1,above_g1_m=max(hmean-link.g1,0),runtime_area_m2=float(m._A_ik[link.id]),clipped_formula_area_m2=expected,area_exact_with_clipped_formula=bool(abs(m._A_ik[link.id]-expected)<1e-8)))
 samples.append(dict(**row,raw_sections={x:s['baseData']['sections'][x] for x in dict.fromkeys([a,b])},adjacent_links=adj))
out=dict(source_edge_points=len(records),unique_channels=len(set(x['channel'] for x in records)),shape_counts={shape:sum(x['shape']==shape for x in records) for shape in sorted(set(x['shape'] for x in records))},all_raw_endpoints_equal=all(x['both_raw_endpoints_equal'] for x in records),max_source_vs_converted_top_diff_m=max(abs(x['source_vs_converted_top_diff_m']) for x in records),worst_three_channels=samples,all_records=records,
 interpretation='输入轮廓覆盖高程不足/计算水位越出其范围的诊断标记；没有堤顶或顶板语义，不能直接称现场漫顶或真实安全限值超标。')
(O/'断面边缘语义核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['all_records','worst_three_channels']},ensure_ascii=False,indent=2));print(json.dumps([dict(channel=x['channel'],adjacent_links=x['adjacent_links']) for x in samples],ensure_ascii=False,indent=2))

B2=D.parent/'26_冻结基线与后续归因/B02_渠池归属修复后的参数同化诊断基线'
all_edges=[]
for ch,meta in c['channel_discretization'].items():
 if ch not in s['baseData']['channels']:continue
 for p in meta['sections']:
  a,b=p['source_interval'];sa=s['baseData']['sections'][a]['yz'];sb=s['baseData']['sections'][b]['yz'];edge=(1-p['fraction'])*min(sa[0][1],sa[-1][1])+p['fraction']*min(sb[0][1],sb[-1][1])
  all_edges.append((ch,p['junction_id'],edge))
history=[]
target_ids={a['junction'] for a in records}
for label,path in [('B02六小时',B2/'六小时诊断起点/最新完整集合.npz'),('B02三十天末',B2/'同化运行/末态集合.npz')]:
 with np.load(path) as state:hh=np.mean([state[f'm{i}_h_Ik'] for i in range(50)],axis=0)
 exceeds=[dict(channel=ch,junction=int(j),above_model_section_edge_m=float(hh[j]+m._z_inv_Ik[j]-edge)) for ch,j,edge in all_edges if hh[j]+m._z_inv_Ik[j]-edge>1e-6]
 history.append(dict(case=label,checked=len(all_edges),above_count=len(exceeds),max_above_m=max([x['above_model_section_edge_m'] for x in exceeds],default=0),overlap_target_68=sum(x['junction'] in target_ids for x in exceeds),details=exceeds))
height_audit=c['section_geometry_audit']
out['baseline_mean_state_checks']=history
out['open_channel_g1_source_audit']=dict(link_count=len(height_audit),all_output_g1_equals_target_section_height=all(abs(x['output_height']-x['target_height'])<1e-12 for x in height_audit),note='连接g1为相邻计算断面的中点插值几何高度，单位m；不是绝对高程，也不是minimum_initial_depth_m。目标高度来自yz最大最小高差及safe插值。')
(O/'断面边缘语义核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(dict(baseline=[{k:v for k,v in x.items() if k!='details'} for x in history],height_audit=out['open_channel_g1_source_audit']),ensure_ascii=False,indent=2))
