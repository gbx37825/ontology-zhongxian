from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=R/'19_水位精度根因';checks=[]
for old,new in [('A02_C07并联物理转换30天','A02R_C07物理转换自适应30天'),('A03_C07物理转换加观测初态30天','A03R_C07物理加初态自适应30天')]:
 a=pd.read_csv(R/'06_结果'/old/'逐闸水位流量.csv');b=pd.read_csv(R/'06_结果'/new/'逐闸水位流量.csv');j=a.merge(b,on=['station_code','time_s'],suffixes=('_a','_b'));diff=max(float(abs(j[c+'_a']-j[c+'_b']).max()) for c in ['forebay_m','afterbay_m','flow_m3s']);assert diff<1e-10;checks.append(dict(failed=old,retry=new,common_output_max_difference_before_failure=diff,rows_compared=len(j)))
(O/'减半步长失败前一致性.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
stats=pd.read_csv(O/'五组单因素精度汇总.csv');runs=['S02_C07输入及闸流修复严格30天','A01_C07观测初态30天','A02R_C07物理转换自适应30天','A03R_C07物理加初态自适应30天','A04_C07物理初态原JSON闸系数30天'];labels=['稳定对照','仅观测初态','仅倒虹吸物理转换','物理转换＋观测初态','再直接使用JSON闸系数'];obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');obs=obs[obs.valid&obs.pdf_comparable&(obs.metric!='flow_m3s')&(obs.time_s>0)]
fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
v=stats[(stats.period=='全30天')&(stats.area=='作者88侧')&(stats.basis=='作者修正')].set_index('run').loc[runs].mae.to_numpy();axes[0].barh(labels,v,color=['#7c8993','#3982a3','#cf9842','#267658','#ae6262']);axes[0].invert_yaxis();axes[0].set_xlabel('30天水位MAE（m），相同作者88侧');
for i,value in enumerate(v):axes[0].text(value+.006,i,f'{value:.3f}',va='center')
axes[0].set_xlim(0,.68)
for run,label in zip(runs[:4],labels[:4]):
 d=pd.read_csv(R/'06_结果'/run/'逐闸水位流量.csv').melt(id_vars=['station_code','time_s'],var_name='metric',value_name='sim').merge(obs,on=['station_code','time_s','metric']);d['error']=abs(d.sim-d.value);d['day']=np.ceil(d.time_s/86400).astype(int);by=d.groupby('day').error.mean();axes[1].plot(by.index,by.values,label=label)
axes[1].set(xlabel='模拟日',ylabel='逐日水位MAE（m）');axes[1].legend();axes[1].grid(alpha=.2);fig.savefig(O/'初态与物理转换根因对照.png',dpi=210);fig.savefig(O/'初态与物理转换根因对照.svg');plt.close(fig)
summary=json.loads((O/'误差结构与点位摘要.json').read_text());summary['single_factor_table']=stats[(stats.period=='全30天')&(stats.area=='作者88侧')&(stats.basis=='作者修正')].to_dict('records');summary['input_sha256']=hashlib.sha256((R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json').read_bytes()).hexdigest();summary['parallel_hook_sha256']=hashlib.sha256((R/'04_脚本/并联结构物理修复.py').read_bytes()).hexdigest();summary['position_evidence_notes']='位置统计优先；两份安装表副本与上轮核对文件哈希一致；历史观测列的具体仪器身份仍未知'
(O/'本轮根因证据总索引.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(checks)
