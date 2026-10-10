from 试验水力工具 import *
O=Path(__file__).resolve().parent;e,s,c,f,params=build();m=e.model;rows=[]
for path in sorted(O.glob('*_状态.npz')):
 with np.load(path) as z:
  H=z['H_j'];h=z['h_Ik'];Q=z['Q_o'];finite=all(np.isfinite(z[k]).all() for k in STATE_KEYS)
  exceed=[];count=0
  for name,meta in c['channel_discretization'].items():
   if name not in s['baseData']['channels']:continue
   for p in meta['sections']:
    a,b=p['source_interval']
    if a not in s['baseData']['sections'] or b not in s['baseData']['sections']:continue
    za=s['baseData']['sections'][a]['yz'];zb=s['baseData']['sections'][b]['yz'];fraction=p['fraction'];edge=(1-fraction)*min(za[0][1],za[-1][1])+fraction*min(zb[0][1],zb[-1][1]);j=p['junction_id'];diff=float(h[j]+m._z_inv_Ik[j]-edge);count+=1
    if diff>1e-6:exceed.append(dict(channel=name,junction=int(j),above_model_section_edge_m=diff))
  if path.name=='目标时刻恒边界稳态_状态.npz':(O/'目标稳态超断面边缘明细.json').write_text(json.dumps(exceed,ensure_ascii=False,indent=2))
  rows.append(dict(case=path.stem,finite=finite,min_internal_depth_m=float(h.min()),min_gate_flow_m3s=float(Q.min()),negative_depths=int((h<-1e-8).sum()),model_section_points_checked=count,above_model_section_edge_count=len(exceed),max_above_edge_m=max([x['above_model_section_edge_m'] for x in exceed],default=0),note='输入断面边缘高程筛查，不等于已独立核实工程堤顶或安全超高'))
(O/'稳态物理范围核查.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('核查完成',len(rows),'个稳态')
