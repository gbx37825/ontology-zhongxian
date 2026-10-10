from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;B=R/'26_冻结基线与后续归因';frozen=json.loads((B/'冻结文件校验.json').read_text());datasets=[];stats=[];samples=[]
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
for code,folder,phase,title in [('B01','B01_允许修改水位的状态同化参考基线','analysis_after_update','我们的改水位同化：橙线已使用当前实测修正'),('B02','B02_渠池归属修复后的参数同化诊断基线','forecast_before_update','我们的改参数同化：橙线用此前更新的参数继续计算')]:
 pieces=[]
 for role in ['同化运行','匹配无同化对照']:
  p=B/folder/role/'预报与分析逐点.csv';assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p.relative_to(B))]['sha256'];a=pd.read_csv(p);a=a[(a.station=='JZZ21')&(a.metric=='forebay_m')&(a.operator=='闸端全线')&(a.phase==(phase if role=='同化运行' else 'forecast_before_update'))].sort_values('time_s');assert len(a)==1440;pieces.append(a)
 a,c=pieces;assert np.array_equal(a.time_s,c.time_s) and np.array_equal(a.observed,c.observed)
 datasets.append(dict(code=code,title=title,data=pd.DataFrame(dict(time_s=a.time_s.to_numpy(),observed=a.observed.to_numpy(),model=c.sim.to_numpy(),assimilation=a.sim.to_numpy()))))
datasets.append(dict(code='WXQ',title='WXQ原PDF：此测点原模型反而更贴近实测（曲线估读）',data=pd.read_csv(O/'WXQ_JZZ21闸前曲线估读.csv')))
wxqerrors={}
for key,file in [('model','作者PDF确定性误差同网格估计.csv'),('assimilation','作者PDF同网格误差估计.csv')]:
 f=pd.read_csv(R/'07_对比分析'/file);wxqerrors[key]=f[(f.station_code=='JZZ21')&(f.metric=='forebay_m')&(f.time_s>0)].sort_values('time_s').error_estimate.to_numpy()
for item in datasets:
 d=item['data'];d.to_csv(O/(item['code']+'_JZZ21闸前三线数据.csv'),index=False)
 for kind in ['model','assimilation']:
  err=wxqerrors[kind] if item['code']=='WXQ' else (d[kind]-d.observed).to_numpy();stats.append(dict(source=item['code'],curve=kind,mae_m=float(abs(err).mean()),max_abs_m=float(abs(err).max()),bias_m=float(err.mean()),count_le_0p1=int((abs(err)<=.1).sum()),n=len(err)))
 for day in [5,15,25,30]:
  q=d[d.time_s==86400*day].iloc[0];samples.append(dict(source=item['code'],day=day,date=str(pd.Timestamp('2025-07-22 12:00')+pd.Timedelta(days=day)),observed=q.observed,model=q.model,assimilation=q.assimilation,model_error=q.model-q.observed,assimilation_error=q.assimilation-q.observed))
S=pd.DataFrame(stats);pd.DataFrame(samples).to_csv(O/'示例时刻读数.csv',index=False);S.to_csv(O/'JZZ21闸前精度.csv',index=False)
low=min(item['data'][['observed','model','assimilation']].min().min() for item in datasets)-.04;high=max(item['data'][['observed','model','assimilation']].max().max() for item in datasets)+.19
fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True);fig.subplots_adjust(left=.085,right=.975,bottom=.10,top=.87,hspace=.30)
fig.suptitle('只看 JZZ21 丈八沟闸前｜三条线怎么读',fontsize=20,y=.975);fig.text(.085,.925,'先选同一时刻，再看蓝线、橙线到黑线的竖向距离：距离越小，该时刻水位误差越小。',fontsize=12,color='#334155')
for ax,item in zip(axes,datasets):
 d=item['data'];x=d.time_s/86400;ax.plot(x,d.model,color='#2879bd',lw=1.05,label='模型（无同化）');ax.plot(x,d.assimilation,color='#e77b24',lw=1.15,label='同化');ax.plot(x,d.observed,color='#212529',ls='--',lw=1.0,label='实测',zorder=4)
 ax.set_title(item['code']+'  '+item['title'],loc='left',fontsize=11.5,pad=8);ax.set_xlim(0,30);ax.set_ylim(low,high);ax.grid(color='#dce3eb',lw=.6);ax.set_ylabel('闸前水位高程（m）');ax.legend(loc='upper right',ncol=3,fontsize=9,framealpha=.95);ax.set_xticks(range(0,31,5));ax.tick_params(labelbottom=True)
 q=d[d.time_s==1296000].iloc[0];ax.axvline(15,color='#64748b',lw=.8,ls=':');ax.scatter([15]*3,[q.model,q.assimilation,q.observed],s=23,c=['#2879bd','#e77b24','#212529'],zorder=6)
 es=[q.model-q.observed,q.assimilation-q.observed];text='第15天：模型'+('高' if es[0]>0 else '低')+f'{abs(es[0])*100:.1f}cm；同化'+('高' if es[1]>0 else '低')+f'{abs(es[1])*100:.1f}cm'
 ax.text(.015,.93,text,transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.94),color='#334155')
 vals=S[S.source==item['code']].set_index('curve').mae_m*100
 ax.text(.015,.80,f'30天MAE：模型 {vals["model"]:.2f}cm / 同化 {vals["assimilation"]:.2f}cm',transform=ax.transAxes,va='top',fontsize=10,color='#334155',bbox=dict(facecolor='white',edgecolor='none',alpha=.94))
 for sp in ['top','right']:ax.spines[sp].set_visible(False)
axes[-1].set_xlabel('从2025-07-22 12:00起的模拟天数；第15天＝08-06 12:00')
fig.text(.085,.045,'三幅图仅比较各自蓝线与橙线的变化；模型及同化口径不同，不作跨方法公平排名。WXQ原图闸前同化为青色，此处统一为橙色。',fontsize=8.5,color='#64748b');fig.savefig(O/'JZZ21闸前三线读图.png',dpi=190);fig.savefig(O/'JZZ21闸前三线读图.svg');plt.close(fig);print(S.to_string(index=False));print(pd.DataFrame(samples).query('day==15').to_string(index=False))
