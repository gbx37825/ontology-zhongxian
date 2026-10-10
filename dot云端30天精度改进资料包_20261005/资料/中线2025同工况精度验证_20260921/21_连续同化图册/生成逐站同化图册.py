from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1];D=R/'20_连续状态同化';O=Path(__file__).resolve().parent
for name in ['E01_20成员30天连续水位同化','E02_同集合30天无同化对照']:
 state=json.loads((D/name/'进度.json').read_text());assert state['status']=='COMPLETED' and state['completed_cycles']==1440
ass=pd.read_csv(D/'E01_20成员30天连续水位同化/预报与分析逐点.csv');ctl=pd.read_csv(D/'E02_同集合30天无同化对照/预报与分析逐点.csv');ass=ass[ass.operator=='闸端全线'];ctl=ctl[(ctl.operator=='闸端全线')&(ctl.phase=='forecast_before_update')];ops=json.loads((D/'E01_20成员30天连续水位同化/观测算子与留出.json').read_text());roles={(a['station'],a['metric']):'参与位置同化' if a['assimilated'] else '空间留出' for a in ops};names={a['station_code']:a['source_name'] for a in json.loads((R/'02_输入/2025节制闸站名映射.json').read_text())};stats=pd.read_csv(D/'30天连续同化精度汇总.csv')
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':8,'axes.linewidth':.65,'lines.linewidth':.8})
path=O/'中线30天连续同化逐站水位与误差对比.pdf';metrics=[];stations=sorted(ass.station.unique(),key=lambda x:int(x[3:]));assert len(stations)==59
with PdfPages(path,metadata={'Title':'中线30天连续同化逐站水位与误差对比','Author':'HydroOS 仿真验证','Subject':'2025-07-22 12:00 至 2025-08-21 12:00；20成员EnKF及同集合对照'}) as pdf:
 fig=plt.figure(figsize=(12,5.5));fig.text(.055,.9,'中线30天连续同化｜逐站水位与误差',fontsize=21,weight='bold');fig.text(.055,.825,'2025-07-22 12:00 — 2025-08-21 12:00   ·   59站 / 118侧   ·   20成员',fontsize=11,color='#334155')
 rows=[]
 for scope,label in [('作者可比闸端88侧','作者可比88侧'),('全线闸端118侧','全线118侧'),('同化测点34侧','同化点34侧（位置算子）'),('空间留出9侧','留出点9侧（位置算子）')]:
  def val(run,phase):return stats[(stats.run.str.startswith(run))&(stats.phase==phase)&(stats.scope==scope)].mae.iloc[0]
  rows.append([label,f'{val("E02","forecast_before_update"):.4f}',f'{val("E01","forecast_before_update"):.4f}',f'{val("E01","analysis_after_update"):.4f}'])
 ax=fig.add_axes([.055,.48,.89,.27]);ax.axis('off');table=ax.table(cellText=rows,colLabels=['水位 MAE（m）','无同化对照','同化前预报','同化后分析'],cellLoc='center',loc='center',colWidths=[.4,.2,.2,.2]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
 for (i,j),cell in table.get_celld().items():
  cell.set_edgecolor('#d7dce2');cell.set_facecolor('#e9eff5' if i==0 else '#f7f9fb' if i%2 else 'white')
 notes=[
 '图册结构参照作者 stations_30d_flow.pdf：每站一页，上图水位，下图“模型−实测”误差。',
 '无同化为同初始集合、同过程扰动的对照；预报在使用当次观测之前，分析在更新之后。每30分钟同化一次。',
 '各站图统一取闸端水位；参与同化的实际算子按安装位置插值。页脚标明同化/留出身份，不混用评价口径。',
 '观测层保留原始尖点；黄金河闸前后分别加0.245 m，古运河闸后加0.301 m，与此前核实口径一致。',
 '模型从0.5小时首次输出，误差统计不含t=0。±0.1 m线仅为误差参考，不表示已全站达标。',
 '这是已知历史边界与开度驱动的回放，不是未知未来输入的真实预报。净同化蓄水调整约1,097万 m³。',
 '灰线：实测；蓝/红：无同化闸前/后；青/橙虚线：预报；青/橙实线：分析。所有曲线均为本次实际结果。']
 for i,t in enumerate(notes):fig.text(.06,.405-i*.042,t,fontsize=8.5,color='#334155')
 fig.text(.95,.045,'说明页  ·  后续59页按JZZ2—JZZ60排列',ha='right',fontsize=8,color='#64748b');pdf.savefig(fig);plt.close(fig)
 for page,st in enumerate(stations,1):
  fig,(ax,er)=plt.subplots(2,1,figsize=(12,5.5),sharex=True,gridspec_kw={'height_ratios':[2.2,1]});fig.subplots_adjust(left=.075,right=.98,bottom=.15,top=.865,hspace=.12)
  fig.suptitle(f'{st} {names.get(st,st)}   （30天连续同化）',y=.966,fontsize=12)
  fig.text(.075,.908,'2025-07-22 12:00 — 08-21 12:00',fontsize=8,color='#475569')
  roletext='；'.join(f'{label}：{roles.get((st,met),"未参与时变同化")}' for met,label in [('forebay_m','闸前'),('afterbay_m','闸后')]);fig.text(.98,.908,roletext,ha='right',fontsize=8,color='#475569')
  metricparts=[]
  for met,side,basecolor,acolor in [('forebay_m','闸前','#2877c7','#00aabd'),('afterbay_m','闸后','#da4242','#f28b2c')]:
   f=ass[(ass.station==st)&(ass.metric==met)&(ass.phase=='forecast_before_update')].sort_values('time_s');a=ass[(ass.station==st)&(ass.metric==met)&(ass.phase=='analysis_after_update')].sort_values('time_s');b=ctl[(ctl.station==st)&(ctl.metric==met)].sort_values('time_s');assert len(f)==len(a)==len(b)==1440;assert np.array_equal(f.time_s,a.time_s) and np.array_equal(f.time_s,b.time_s);assert np.allclose(f.observed,b.observed)
   x=f.time_s.to_numpy()/86400;y=f.observed.to_numpy();ax.plot(x,y,color='#858585',ls='-' if met=='forebay_m' else '--',lw=.8,label='实测 '+side,zorder=4)
   vals=[('无同化',b.sim.to_numpy(),basecolor,'-',.75),('预报',f.sim.to_numpy(),acolor,'--',.7),('分析',a.sim.to_numpy(),acolor,'-',.9)]
   valsmae=[]
   for label,v,col,style,width in vals:
    ax.plot(x,v,color=col,ls=style,lw=width,label=label+' '+side);err=v-y;er.plot(x,err,color=col,ls=style,lw=width,label=label+' '+side);mae=float(abs(err).mean());valsmae.append(mae);metrics.append(dict(station=st,side=met,phase=label,n=len(err),mae=mae,max_error=float(abs(err).max()),pdf_comparable=bool(f.pdf_comparable.iloc[0])))
   metricparts.append(f'{side} MAE：'+ ' / '.join(f'{z:.3f}' for z in valsmae))
  ax.set_ylabel('水位（m）');ax.legend(loc='upper left',ncol=4,fontsize=6.5,framealpha=.9,borderpad=.3,columnspacing=1.2);ax.margins(x=0,y=.15)
  er.axhspan(-.1,.1,color='#e7f2e8',zorder=0);er.axhline(0,color='#222',lw=.55);er.axhline(.1,color='#bf3131',ls='-.',lw=.7);er.axhline(-.1,color='#824b9b',ls='-.',lw=.7)
  er.annotate('上限 +0.1 m',xy=(1,.1),xycoords=('axes fraction','data'),xytext=(-3,2),textcoords='offset points',ha='right',va='bottom',fontsize=6,color='#a82626');er.annotate('下限 −0.1 m',xy=(1,-.1),xycoords=('axes fraction','data'),xytext=(-3,-2),textcoords='offset points',ha='right',va='top',fontsize=6,color='#753f8e')
  er.set_ylabel('模型−实测（m）');er.set_xlabel('时间（d）');er.legend(loc='lower left',ncol=3,fontsize=6.2,framealpha=.85,borderpad=.25);er.set_xlim(0,30);er.set_xticks(range(0,31,5));er.margins(y=.2)
  for aax in [ax,er]:aax.grid(color='#dedede',lw=.5,alpha=.8);aax.tick_params(labelsize=8)
  fig.text(.075,.055,'  ｜  '.join(metricparts)+' m   （顺序：无同化 / 预报 / 分析）',fontsize=7.4,color='#334155');fig.text(.98,.025,f'{page}/59',ha='right',fontsize=7,color='#64748b');pdf.savefig(fig);plt.close(fig)
pd.DataFrame(metrics).to_csv(O/'图册逐页误差核验.csv',index=False);(O/'图册生成清单.json').write_text(json.dumps(dict(pages=60,stations=59,points_per_side=1440,pdf=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_runs=['E01','E02'],layout_reference='stations_30d_flow.pdf'),ensure_ascii=False,indent=2));print(path.resolve())
