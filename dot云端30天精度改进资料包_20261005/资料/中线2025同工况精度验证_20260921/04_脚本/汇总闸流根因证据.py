from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=R/'11_Pipedream振荡根因'
font=Path('/System/Library/Fonts/STHeiti Light.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
cases=[]
for p in sorted(O.glob('*_iter*.json')):
 d=json.loads(p.read_text());d['evidence']=p.name;d.setdefault('station','JZZ21' if 'JZZ21' in p.stem else 'JZZ42');d.setdefault('physics','legacy' if 'legacy' in p.stem else 'physical');d['tracked_gate']=('JZZ42-七里河倒虹吸出口节制闸-3' if d['station']=='JZZ42' else 'JZZ21-丈八沟倒虹吸出口节制闸-4');d['only_gate']=d['tracked_gate'] if 'only' in d['case'] else None
 if 'max_jzz42_mismatch' in d:d['max_station_mismatch']=d.pop('max_jzz42_mismatch')
 d['convergence_count_interpretation']='单次求解仅记录步间水位变化，不能把该计数解释为迭代失败' if d['iterations']==1 else '最后两次迭代节点水位差>=0.0015m的步数'
 cases.append(d)
pd.DataFrame(cases).to_csv(O/'对照试验汇总.csv',index=False)
diffs=[]
for da,db in [(15,7.5),(7.5,3.75)]:
 a=pd.read_csv(O/f'integral_tangent_iter20_dt{da:g}.csv');b=pd.read_csv(O/f'integral_tangent_iter20_dt{db:g}.csv');a.time_s=a.time_s.astype(float);b.time_s=b.time_s.astype(float);d=a.merge(b,on='time_s',suffixes=('_a','_b'));diffs.append(dict(dt_pair=[da,db],**{k:float(abs(d[k+'_a']-d[k+'_b']).max()) for k in ['forebay_m','afterbay_m','gate_flow']}))
# 原数值窗重放与已冻结结果完全重合。
a=pd.read_csv(O/'original_iter1_dt15.csv');b=pd.read_csv(R/'06_结果/PHV00_并联物理调闸步长窗/dt_15s_JZZ42.csv');d=a.merge(b,on='time_s',suffixes=('_a','_b'));replay={k:float(abs(d[k+'_a']-d[k+'_b']).max()) for k in ['forebay_m','afterbay_m','gate_flow']};assert max(replay.values())<1e-10
files=[R/'02_输入/C00_实际开度负值修复_2025同工况30天.json',R/'02_输入/C01_目标开度独立假设_2025同工况30天.json',R/'04_脚本/闸流振荡根因核验.py',R/'04_脚本/并联结构物理修复.py',R/'09_运行源码/pipedream_solver/nsuperlink.py',R/'09_运行源码/pipedream_solver/superlink.py',R/'09_运行源码/wxq_pipedream/wxq_to_superlink.py']
manifest=dict(input_or_production_changes=False,scope='定位根因的局部反事实试验，不是正式率定或30天精度验收',cases=cases,dt_comparison=diffs,frozen_replay_difference=replay,sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(O/'根因证据清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
fig,ax=plt.subplots(3,1,figsize=(11,11),constrained_layout=True)
a=pd.read_csv(O/'original_iter20_dt15_legacy.csv');b=pd.read_csv(O/'integral_only42_tangent_iter20_dt15_legacy.csv')
ax[0].plot(a.time_s/3600,a.gate_flow,color='#be3434',lw=.8,label='原 Pipedream（20次迭代）');ax[0].plot(b.time_s/3600,b.gate_flow,color='#176e94',lw=1.5,label='仅第三孔连续闸流诊断');ax[0].set(title='JZZ42：同一4小时初态、同一实际开度，原基线局部重放',ylabel='全站过闸流量（m³/s）',xlabel='自起报时刻起（小时）');ax[0].legend()
d=pd.read_csv(O/'定水位开度扫描_geometry.csv');depth=d.depth.iloc[0];opening=np.linspace(depth-.05,depth+.05,1000);hu=d.hu.iloc[0];hd=d.hd.iloc[0];width=d.width.iloc[0]
C=d.q_original.iloc[1]/(width*np.sqrt(2*9.81*(hu-hd)))
area=width*np.minimum(opening,depth);q=C*area*np.sqrt(2*9.81*np.where(opening<depth,hu-hd,depth));lo=depth-(hu-hd);top=np.minimum(opening,depth);wet=np.minimum(top,lo);smooth=C*width*np.sqrt(2*9.81)*(wet*np.sqrt(hu-hd)+(2/3)*(np.maximum(depth-wet,0)**1.5-np.maximum(depth-top,0)**1.5))
ax[1].plot(opening,q,color='#be3434',label='原分段公式');ax[1].plot(opening,smooth,color='#176e94',label='连续关系（诊断用）');ax[1].axvline(depth,color='gray',ls=':');ax[1].set(title='固定闸前83.900671m、闸后83.527682m：开度跨水面时流量跳变',xlabel='第三孔开度（m）',ylabel='单孔流量（m³/s）');ax[1].legend()
a=pd.read_csv(O/'original_iter20_dt15_JZZ21.csv');b=pd.read_csv(O/'integral_only21_tangent_iter20_dt15_JZZ21.csv')
ax[2].plot((a.time_s-2332800)/3600,a.gate_flow,color='#be3434',lw=.7,label='原公式（20次迭代）');ax[2].plot((b.time_s-2332800)/3600,b.gate_flow,color='#176e94',lw=1.5,label='仅第四孔连续闸流诊断');ax[2].set(title='JZZ21：第28天另一个异常事件，同初态、同目标开度假设',xlabel='第28天起算（小时）',ylabel='全站过闸流量（m³/s）');ax[2].legend()
for x in ax:x.grid(alpha=.15)
fig.savefig(O/'闸流振荡根因对照.png',dpi=220);fig.savefig(O/'闸流振荡根因对照.svg');plt.close(fig)
print(json.dumps({'replay_difference':replay,'dt_comparison':diffs,'cases':len(cases)},ensure_ascii=False))
