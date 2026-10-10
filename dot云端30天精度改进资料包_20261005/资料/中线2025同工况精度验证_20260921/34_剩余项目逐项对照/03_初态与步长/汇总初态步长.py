from pathlib import Path
import json,numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];obs=pd.read_csv(R/'26_冻结基线与后续归因/共享输入/作者已核实修正观测层.csv');obs=obs[obs.valid];names=['I00_原初态15秒','I01_闸流一致化15秒','I02_原初态7点5秒'];data={};rows=[];diffs=[]
for name in names:
 d=O/name;result=json.loads((d/'结果.json').read_text());assert result['status']=='COMPLETED' and result['periods']==144;f=pd.read_csv(d/'水位.csv');assert len(f)==59*144 and not f.duplicated(['station','time_s']).any();data[name]=f
 with np.load(d/'末态.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
 a=f.melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['station_code','metric','time_s','value','pdf_comparable']],left_on=['station','metric','time_s'],right_on=['station_code','metric','time_s'],validate='one_to_one');a['error']=a.sim-a.value
 for period,l,r in [('首6小时',0,21600),('6至24小时',21600,86400),('第2天',86400,172800),('第3天',172800,259200),('全3天',0,259200)]:
  q=a[(a.time_s>l)&(a.time_s<=r)]
  for scope,g in [('全线118个水位测点',q),('作者88个水位测点',q[q.pdf_comparable])]:rows.append(dict(run=name,period=period,scope=scope,n=len(g),mae=abs(g.error).mean(),rmse=np.sqrt((g.error**2).mean()),max_abs=abs(g.error).max()))
base=data[names[0]]
for name in names[1:]:
 f=data[name];assert np.array_equal(base[['station','time_s']],f[['station','time_s']]);dh=(f[['forebay_m','afterbay_m']]-base[['forebay_m','afterbay_m']]).to_numpy();diffs.append(dict(run=name,mean_abs_head_difference=float(abs(dh).mean()),max_abs_head_difference=float(abs(dh).max()),median_abs_head_difference=float(np.median(abs(dh)))))
pd.testing.assert_frame_equal(base,data[names[1]],check_exact=True)
with np.load(O/names[0]/'末态.npz') as a,np.load(O/names[1]/'末态.npz') as b:assert all(np.array_equal(a[k],b[k]) for k in a.files)
s=pd.DataFrame(rows);s.to_csv(O/'初态与步长精度对照.csv',index=False);pd.DataFrame(diffs).to_csv(O/'相对原轨迹变化.csv',index=False);print(s[s.period=='全3天'].to_string(index=False));print(diffs)
result=dict(initial_Qo_sync_trajectory_exact_equal=True,initial_Qo_sync_final_arrays_exact_equal=True,head_and_channel_Q_initially_unchanged=True,runs=names,differences=diffs,note='只改初始代数Q_o与原轨迹完全相同，说明该快照差被首步代数重算消化，不能将47m3/s快照差直接当水位偏差因果证据；其他渠内H/Q动态初态准确性尚未由本实验证明。')
(O/'核验结论.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
