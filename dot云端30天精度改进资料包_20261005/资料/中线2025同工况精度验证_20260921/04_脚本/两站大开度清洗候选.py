from pathlib import Path
import copy,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'16_两站分别清洗复跑';O.mkdir(exist_ok=True);base=R/'14_输入修补交付/C03_含七里河重点脉冲修补候选_2025同工况30天.json';beforehash=hashlib.sha256(base.read_bytes()).hexdigest();data=json.loads(base.read_text());m=next(iter(data.values()));raw=pd.read_csv(R/'12_全量实测质量核查/五个重点站30天开度换算米.csv.gz',parse_dates=['时间']);start=pd.Timestamp('2025-07-22 12:00');changes=[];segments=[]
for st in ['JZZ27','JZZ23']:
 g=raw[raw.模型站号==st].sort_values('时间').reset_index(drop=True);assert len(g)==1441
 for key,(typ,name) in m['timeSeries']['seriesLink'].items():
  if typ!='gates' or not name.startswith(st+'-'):continue
  hole=int(name.rsplit('-',1)[1]);a=g[f'孔{hole}实际开度'].to_numpy(float);target=g[f'孔{hole}目标开度'].to_numpy(float);large=(a>=6)&(target<=.1)&(target>=0);bad=((a>.1)&(target<=.1)&(target>=0))|~np.isfinite(a)|(a<0);ids=np.flatnonzero(bad);groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(ids) else []
  for ix in groups:
   if not large[ix].any():continue
   l=ix[0]-1;r=ix[-1]+1;assert l>=0 and r<len(a);assert 0<=a[l]<=.1 and 0<=a[r]<=.1,(name,l,r,a[l],a[r]);assert (target[l:r+1]>=0).all() and (target[l:r+1]<=.1).all();assert (g.时间.iloc[r]-g.时间.iloc[l]).total_seconds()<=4*3600
   segid=f'{st}-{hole}-{ix[0]}';segments.append(dict(id=segid,station=st,hole=hole,start=str(g.时间.iloc[ix[0]]),end=str(g.时间.iloc[ix[-1]]),large_raw_points=int(large[ix].sum()),bridged_invalid_points=int((~np.isfinite(a[ix])|(a[ix]<0)).sum()),connected_partial_points=int(((a[ix]>=0)&~large[ix]).sum()),left=str(g.时间.iloc[l]),right=str(g.时间.iloc[r]),left_value=a[l],right_value=a[r]))
   for i in ix:
    value=float(a[l]+(a[r]-a[l])*(i-l)/(r-l));old=float(m['timeSeries']['seriesData'][key][i][1]);changes.append(dict(segment=segid,station=st,hole=hole,name=name,key=key,index=int(i),timestamp=str(g.时间.iloc[i]),source_row=int(g.源行号.iloc[i]),raw_m=float(a[i]),C03_value_m=old,replacement_m=value,changed_from_C03=old!=value,reason='用户确认该类突然大开度为坏数据；目标保持关闭，前后实测近零' if large[i] else ('与大开度坏段相邻的原负码，联合重插值，避免保留旧插值造成的虚假开度' if not np.isfinite(a[i]) or a[i]<0 else '同一大开度异常连续段的中间或回落值，目标仍保持关闭')))
variants=[('C04_仅济河清洗',['JZZ27']),('C05_仅金水河清洗',['JZZ23']),('C06_济河金水河合并清洗',['JZZ27','JZZ23'])];result=[]
for name,sts in variants:
 obj=copy.deepcopy(data);mm=next(iter(obj.values()));used=[x for x in changes if x['station'] in sts]
 for x in used:mm['timeSeries']['seriesData'][x['key']][x['index']][1]=x['replacement_m']
 diff=[]
 for k,ar in mm['timeSeries']['seriesData'].items():
  aa=np.asarray(ar,float);bb=np.asarray(m['timeSeries']['seriesData'][k],float);assert aa.shape==(1441,2) and np.isfinite(aa).all() and np.array_equal(aa[:,0],bb[:,0]);diff.extend((k,int(i)) for i in np.flatnonzero(aa[:,1]!=bb[:,1]))
 assert set(diff)=={(x['key'],x['index']) for x in used if x['changed_from_C03']}
 for k in m:
  if k!='timeSeries':assert mm[k]==m[k]
 p=O/(name+'_2025同工况30天.json');p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');result.append(dict(version=name,path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),new_changed_points=len(diff),treated_raw_points=len(used),stations=sts))
pd.DataFrame(changes).to_csv(O/'逐点清洗与补齐清单.csv',index=False);pd.DataFrame(segments).to_csv(O/'异常连续段与补值端点.csv',index=False);assert hashlib.sha256(base.read_bytes()).hexdigest()==beforehash
manifest=dict(baseline=str(base.resolve()),baseline_sha256=beforehash,versions=result,segments=len(segments),raw_large_points=int(sum(x['large_raw_points'] for x in segments)),bridged_invalid_points=int(sum(x['bridged_invalid_points'] for x in segments)),connected_partial_points=int(sum(x['connected_partial_points'] for x in segments)),preserved='丈八沟及其他站、评分观测、原归一化高度、糙率、系数、初态、求解器均不改',user_decision='按用户确认将从小开度突然跳到6—7m且目标保持关闭的记录视作坏数据；不将已有大开度并有目标支持的正常动作一概删除')
(O/'清洗输入核验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
