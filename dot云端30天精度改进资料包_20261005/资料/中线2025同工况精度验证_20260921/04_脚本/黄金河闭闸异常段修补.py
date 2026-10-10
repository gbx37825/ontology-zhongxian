from pathlib import Path
import copy,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'17_求解器修复';base=R/'16_两站分别清洗复跑/C06_济河金水河合并清洗_2025同工况30天.json';data=json.loads(base.read_text());m=next(iter(data.values()));old=copy.deepcopy(m);g=pd.read_csv(R/'12_全量实测质量核查/五个重点站30天开度换算米.csv.gz',parse_dates=['时间']);g=g[g.模型站号=='JZZ9'].sort_values('时间').reset_index(drop=True);a=g.孔3实际开度.to_numpy();t=g.孔3目标开度.to_numpy();bad=((a>.1)&(t>=0)&(t<=.1))|(a<0)|~np.isfinite(a);ix=np.flatnonzero(bad);segments=np.split(ix,np.where(np.diff(ix)>1)[0]+1);key=next(k for k,(typ,name) in m['timeSeries']['seriesLink'].items() if typ=='gates' and name.startswith('JZZ9-') and name.endswith('-3'));changes=[];segs=[]
for ids in segments:
 if np.nanmax(a[ids])<6:continue
 l=ids[0]-1;r=ids[-1]+1;assert 0<=a[l]<=.1 and 0<=a[r]<=.1 and np.all(t[l:r+1]==0)
 hours=(g.时间.iloc[r]-g.时间.iloc[l]).total_seconds()/3600;mode='长段闭闸重建假设：目标连续为0，端点为近零实测，不作短缺口确认' if hours>4 else '短段同孔有效端点插值'
 segs.append(dict(start=str(g.时间.iloc[ids[0]]),end=str(g.时间.iloc[ids[-1]]),bracket_hours=hours,points=len(ids),left_value=float(a[l]),right_value=float(a[r]),method=mode))
 for i in ids:
  v=float(a[l]+(a[r]-a[l])*(i-l)/(r-l));before=m['timeSeries']['seriesData'][key][i][1];m['timeSeries']['seriesData'][key][i][1]=v;changes.append(dict(timestamp=str(g.时间.iloc[i]),source_row=int(g.源行号.iloc[i]),raw_m=float(a[i]),C06_m=before,C07_m=v,index=int(i),method=mode))
allowed={(key,d['index']) for d in changes if d['C06_m']!=d['C07_m']};diff=[]
for k,v in m['timeSeries']['seriesData'].items():
 aa=np.array(v);bb=np.array(old['timeSeries']['seriesData'][k]);assert aa.shape==(1441,2) and np.isfinite(aa).all() and np.array_equal(aa[:,0],bb[:,0]);diff.extend((k,int(i)) for i in np.flatnonzero(aa[:,1]!=bb[:,1]))
assert set(diff)==allowed
for k in m:
 if k!='timeSeries':assert m[k]==old[k]
p=O/'C07_黄金河闭闸重建候选_2025同工况30天.json';p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');pd.DataFrame(changes).to_csv(O/'黄金河逐点修补.csv',index=False);(O/'黄金河修补依据.json').write_text(json.dumps(dict(segments=segs,changed_points=len(diff),input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest()),ensure_ascii=False,indent=2));print(segs,len(diff))
