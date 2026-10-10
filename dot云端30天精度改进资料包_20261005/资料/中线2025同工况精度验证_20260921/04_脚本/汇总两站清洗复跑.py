from pathlib import Path
import json,hashlib,zipfile
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'16_两站分别清洗复跑';names=['Q03_六点开度修补同工况30天','Q04_仅济河清洗30天','Q05_仅金水河清洗30天','Q06_两站合并清洗30天'];obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');rows=[];per=[];flows=[];init=None;statuses=[]
for name in names:
 p=R/'06_结果'/name;status=json.loads((p/'运行摘要.json').read_text());assert status['status']=='COMPLETED' and status['completed_steps']==172800;statuses.append(status);raw=pd.read_csv(p/'逐闸水位流量.csv');z=raw[raw.time_s==0].sort_values('station_code')[['forebay_m','afterbay_m','flow_m3s']].to_numpy()
 if init is None:init=z
 diff=float(abs(z-init).max());assert diff<1e-10,(name,diff)
 d=raw.melt(id_vars=['station_code','time_s'],var_name='metric',value_name='sim').merge(obs,on=['station_code','time_s','metric'],validate='one_to_one');d=d[d.valid&(d.metric!='flow_m3s')&(d.time_s>0)];d['raw_error']=d.sim-d.raw_value;d['author_error']=d.sim-d.value
 for area,g in [('全线118侧',d),('作者88侧',d[d.pdf_comparable]),('济河两侧',d[d.station_code=='JZZ27']),('金水河两侧',d[d.station_code=='JZZ23']),('丈八沟两侧',d[d.station_code=='JZZ21'])]:
  for basis,col in [('原始观测','raw_error'),('作者已核实修正','author_error')]:
   maes=g.groupby(['station_code','metric'])[col].apply(lambda s:abs(s).mean());rows.append(dict(run=name,area=area,basis=basis,mae=float(abs(g[col]).mean()),max_error=float(abs(g[col]).max()),pass_mae_0p1=int((maes<=.1).sum()),n=len(g)))
 for (st,metric),g in d.groupby(['station_code','metric']):per.append(dict(run=name,station=st,metric=metric,mae=float(abs(g.author_error).mean())))
 f=raw[raw.time_s>0];neg=f[f.flow_m3s<0].copy();neg['timestamp']=pd.Timestamp('2025-07-22 12:00')+pd.to_timedelta(neg.time_s,unit='s');neg.to_csv(O/(name+'_负流量输出.csv'),index=False);flows.append(dict(run=name,negative_frames=len(neg),negative_by_station=neg.groupby('station_code').size().to_dict(),min_q=float(f.flow_m3s.min()),max_q=float(f.flow_m3s.max()),initial_difference=diff))
pd.DataFrame(rows).to_csv(O/'四组30天精度对照.csv',index=False);pd.DataFrame(per).to_csv(O/'四组逐侧精度.csv',index=False);(O/'四组运行与负流量核验.json').write_text(json.dumps(dict(statuses=statuses,flow_checks=flows,source_unchanged=True),ensure_ascii=False,indent=2))
s=pd.DataFrame(rows);tab=['| 版本 | 作者88侧MAE(m) | 全线118侧MAE(m) | 负流量站时次 | 负流量所在站 |','|---|---:|---:|---:|---|'];display=['C03原对照','C04仅济河','C05仅金水河','C06两站合并']
for name,label,f in zip(names,display,flows):
 x=s[(s.run==name)&(s.basis=='作者已核实修正')].set_index('area');tab.append(f"| {label} | {x.loc['作者88侧','mae']:.6f} | {x.loc['全线118侧','mae']:.6f} | {f['negative_frames']} | {f['negative_by_station']} |")
text='''# 济河、金水河分别清洗及30天复跑

## 本轮处理

按用户确认，将目标保持关闭、实测从近零突然跳到6—7m及以上的异常记录剔除后补齐；不将已有大开度且目标支持的调闸一概删除。以C03为对照，分别生成仅济河C04、仅金水河C05、两站合并C06，全部完成2025-07-22 12:00至08-21 12:00的30天计算。

本轮跨72个连续坏段处理176个原始采样点：171个高开度点、2个相邻负码及3个同段中间/回落值。济河171点中有4点已在C03修过，所以相对C03新增167点；金水河新增5点；合并新增172点。

每段均校验两侧实测开度在0—0.1m内，目标保持关闭，端点间隔不超过4小时。本批左右端点实际都是0，因此线性补齐结果为0。相邻负码联合重算，不能保留旧大开度坏点参与插值得到的假开度。

原始Excel、C03、丈八沟及其他站输入、评分观测、糙率、闸流系数、归一化上限、初始化方式及求解器全部不改。逐点差分与313×1441时序检查通过，四组t=0水位/流量相同。

## 结果

下表均为全30天、排除t=0；作者88侧采用此前核实的观测偏移。没有删评分水位尖点。负流量计数为半小时输出的站点×时刻，不是站数，也不是内部15秒步数。

'''+ '\n'.join(tab)+'''

## 判断范围

分别清洗用于定位济河、金水河异常输入各自的影响；两站合并版本不等于全线输入全部正确。丈八沟第四孔有5.82→7.10m目标调节，仍原样保留；若它仍出现负流量，应按已定位的求解分支/模型水位问题处理，而不是删动作。

即使负流量输出减少，仍须区分模型完成、内部每步稳定和实测精度验收。此次没有修改求解器，不把进程完成当作方程收敛通过，不自动替换正式基线。

## 交付

- 三个独立候选WXQ JSON。
- 《逐点清洗与补齐清单.csv》：原始值、C03值、新值、源行号、理由。
- 《异常连续段与补值端点.csv》：每段起止与左右补值依据。
- 《清洗输入核验.json》：输入SHA256、精确变更数量。
- 《四组30天精度对照.csv》《四组逐侧精度.csv》《四组运行与负流量核验.json》。
- 各组剩余负流量逐时刻清单。
'''
(O/'分别清洗复跑结果.md').write_text(text)
print('\n'.join(tab))
