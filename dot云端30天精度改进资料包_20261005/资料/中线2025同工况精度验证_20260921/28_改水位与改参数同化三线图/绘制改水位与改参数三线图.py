from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;B=R/'26_冻结基线与后续归因'
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((B/'冻结文件校验.json').read_text());datasets=[];metrics=[];sources=[]
for code,folder,title in [('B01','B01_允许修改水位的状态同化参考基线','20成员 · 改水位同化（修正后）'),('B02','B02_渠池归属修复后的参数同化诊断基线','50成员 · 改参数同化（渠池归属已修复）')]:
 frames=[]
 for role in ['同化运行','匹配无同化对照']:
  p=B/folder/role/'预报与分析逐点.csv';assert sha(p)==manifest[str(p.relative_to(B))]['sha256'];sources.append({'path':str(p),'sha256':sha(p)})
  phase='analysis_after_update' if code=='B01' and role=='同化运行' else 'forecast_before_update'
  f=pd.read_csv(p);f=f[(f.operator=='闸端全线')&(f.phase==phase)].sort_values(['station','metric','time_s']);assert len(f)==118*1440;frames.append(f)
 a,c=frames;assert np.array_equal(a[['station','metric','time_s']],c[['station','metric','time_s']]);assert np.array_equal(a.observed,c.observed)
 datasets.append(dict(code=code,title=title,a=a,c=c))
assert np.array_equal(datasets[0]['a'].observed,datasets[1]['a'].observed)
names={x['station_code']:x['source_name'] for x in json.loads((R/'02_输入/2025节制闸站名映射.json').read_text())};stations=sorted(datasets[0]['a'].station.unique(),key=lambda s:int(s[3:]));limits={}
for st in stations:
 for met in ['forebay_m','afterbay_m']:
  vals=[]
  for d in datasets:
   a=d['a'];c=d['c'];v=a[(a.station==st)&(a.metric==met)];q=c[(c.station==st)&(c.metric==met)];assert len(v)==len(q)==1440;vals.extend([v.sim.to_numpy(),q.sim.to_numpy(),v.observed.to_numpy()])
  lo=min(x.min() for x in vals);hi=max(x.max() for x in vals);pad=max(.06,(hi-lo)*.13);limits[st,met]=(lo-pad,hi+pad)
def panel(ax,d,st,met):
 a=d['a'];c=d['c'];a=a[(a.station==st)&(a.metric==met)];c=c[(c.station==st)&(c.metric==met)];x=a.time_s.to_numpy()/86400;obs=a.observed.to_numpy()
 ax.plot(x,c.sim,color='#2775bb',lw=1.05,label='模型（无同化）');ax.plot(x,a.sim,color='#e77922',lw=1.15,label='同化（改水位后）' if d['code']=='B01' else '同化（改参数）');ax.plot(x,obs,color='#252b35',lw=.95,ls='--',label='实测',zorder=4)
 mae0=float(abs(c.sim.to_numpy()-obs).mean());mae1=float(abs(a.sim.to_numpy()-obs).mean())
 ax.set_title(('闸前水位' if met=='forebay_m' else '闸后水位')+f'    MAE：模型 {mae0:.3f} m / 同化 {mae1:.3f} m',loc='left',fontsize=10.5,pad=9)
 ax.set_xlim(0,30);ax.set_ylim(*limits[st,met]);ax.set_xticks(range(0,31,5));ax.set_ylabel('水位高程（m）');ax.grid(color='#dbe2e8',lw=.55);ax.legend(loc='best',ncol=3,fontsize=8,framealpha=.93)
 for spine in ['top','right']:ax.spines[spine].set_visible(False)
 assert len(ax.lines)==3
 return dict(baseline=d['code'],station=st,metric=met,n=len(a),model_mae=mae0,assimilation_mae=mae1,model_max_error=float(abs(c.sim.to_numpy()-obs).max()),assimilation_max_error=float(abs(a.sim.to_numpy()-obs).max()))
for d in datasets:
 folder=O/d['code'];folder.mkdir(exist_ok=True);svg=folder/'逐站矢量图';svg.mkdir(exist_ok=True);path=O/(d['code']+'_'+('改水位同化修正后' if d['code']=='B01' else '改参数同化')+'_模型同化实测三线图.pdf')
 with PdfPages(path) as pdf:
  fig=plt.figure(figsize=(12,7.2));fig.text(.07,.88,d['code']+'｜模型 · 同化 · 实测',fontsize=25,weight='bold');fig.text(.07,.79,d['title'],fontsize=17,color='#34475a');notes=['2025年7月22日12:00 — 8月21日12:00；30天、59座闸、118个水位测点。','每座闸一页，闸前与闸后分图；每图严格为三条水位曲线。','蓝线：无同化模型；橙线：改水位后的同化结果（B01）或改参数同化运行结果（B02）；黑虚线：实测。','B01使用当前实测修正后的分析水位；B02用此前调整的参数持续计算，当前参数更新不使水位瞬变。','两份图册同一测点采用相同纵轴范围，所有1440个时刻均保留，不平滑、不裁剪异常尖点。','绘图统一采用模型闸端水位；实际仪器安装位置尚有未闭合项，不能混同为精确同位置验证。','实测采用冻结的已核实修正观测层；黄金河闸前后+0.245m、古运河闸后+0.301m。','B01在线修正水位、流量，糙率和闸流系数不变；B02只修正糙率和闸流系数。','已知历史边界回放，不代表未知未来输入的独立预报；基线冻结不等于精度验收。']
  for i,n in enumerate(notes):fig.text(.07,.67-i*.055,n,fontsize=10,color='#425466')
  fig.text(.93,.055,'说明页｜后续按 JZZ2—JZZ60 排列',ha='right',fontsize=9,color='#64748b');pdf.savefig(fig);plt.close(fig)
  for page,st in enumerate(stations,1):
   fig,axes=plt.subplots(2,1,figsize=(12,7.2),sharex=True);fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.13,hspace=.39)
   fig.suptitle(f"{d['code']}  {st} {names.get(st,st)}",fontsize=16,y=.965);fig.text(.09,.91,d['title']+'    |    2025-07-22 12:00 至 08-21 12:00',fontsize=10,color='#475569')
   for ax,met in zip(axes,['forebay_m','afterbay_m']):metrics.append(panel(ax,d,st,met))
   axes[1].set_xlabel('模拟时间（天）');fig.text(.09,.055,'实测尖点保留。' + ('改水位同化采用修正后分析水位。' if d['code']=='B01' else '改参数同化水位由模型持续计算。') + '历史边界回放。',fontsize=8.5,color='#64748b');fig.text(.97,.055,f'{page}/59',ha='right',fontsize=9,color='#64748b');pdf.savefig(fig);fig.savefig(svg/(st+'.svg'))
   if st in ['JZZ13','JZZ35','JZZ45']:fig.savefig(folder/(st+'_三线图.png'),dpi=240)
   plt.close(fig)
 print('完成',path.name,flush=True)
# 一张沙河四面板便于跨基线直接查看，单面板仍只有三条线。
fig,axes=plt.subplots(2,2,figsize=(15,9));fig.subplots_adjust(left=.065,right=.975,bottom=.11,top=.86,hspace=.40,wspace=.22);fig.suptitle('沙河渡槽进口节制闸｜改水位同化与改参数同化',fontsize=20,y=.97)
for row,d in enumerate(datasets):
 for col,met in enumerate(['forebay_m','afterbay_m']):panel(axes[row,col],d,'JZZ13',met);axes[row,col].set_xlabel('模拟时间（天）' if row else '')
 fig.text(.065,.91-row*.41,d['code']+'  '+d['title'],fontsize=12,weight='bold',color='#34475a')
fig.text(.065,.035,'上排：改水位后的同化结果；下排：改参数同化计算结果。上下排同纵轴；两种同化口径不同。',fontsize=10,color='#64748b');fig.savefig(O/'沙河改水位与改参数三线对照.png',dpi=200);fig.savefig(O/'沙河改水位与改参数三线对照.svg');plt.close(fig)
pd.DataFrame(metrics).to_csv(O/'逐图指标核验.csv',index=False);(O/'绘图来源与校验.json').write_text(json.dumps(dict(source=sources,baselines=2,stations_per_baseline=59,pages_per_pdf=60,panels=236,lines_per_panel=3,points_per_line=1440,phase={'B01':'analysis_after_update','B02':'forecast_before_update'},source_operator='闸端全线',same_ylim_between_baselines=True,output=[dict(path=str(p),sha256=sha(p)) for p in O.glob('*.pdf')]),ensure_ascii=False,indent=2))
