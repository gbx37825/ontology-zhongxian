from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path(__file__).resolve().parent;R=O.parent
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
d=pd.read_csv(O/'WXQ_JZZ21闸前曲线估读.csv');x=d.time_s/86400;errs={}
for name,file in [('模型','作者PDF确定性误差同网格估计.csv'),('同化','作者PDF同网格误差估计.csv')]:
 a=pd.read_csv(R/'07_对比分析'/file);a=a[(a.station_code=='JZZ21')&(a.metric=='forebay_m')&(a.time_s>0)].sort_values('time_s');assert np.array_equal(a.time_s,d.time_s);errs[name]=a.error_estimate.to_numpy()
fig,(ax,er)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[1.7,1]});fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.14,hspace=.20)
fig.suptitle('只看WXQ原PDF：JZZ21丈八沟闸前',fontsize=18,y=.965);fig.text(.09,.90,'保留原图配色；已去掉闸后曲线，避免六条线混在一起。',fontsize=11,color='#475569')
ax.plot(x,d.observed,color='#808080',lw=1.2,label='灰线：实测闸前');ax.plot(x,d.model,color='#1f77b4',lw=1.1,label='蓝线：确定性模型闸前');ax.plot(x,d.assimilation,color='#17becf',lw=1.3,label='青线：EnKF同化闸前');ax.set_ylabel('水位高程（m）');ax.legend(loc='upper left',ncol=3,fontsize=9);ax.margins(y=.25)
er.axhspan(-.1,.1,color='green',alpha=.10);er.axhline(0,color='#222',lw=.8);er.axhline(.1,color='#aa3333',ls='-.',lw=.8);er.axhline(-.1,color='#824b9b',ls='-.',lw=.8);er.plot(x,errs['模型'],color='#1f77b4',lw=1.1,label='模型−实测');er.plot(x,errs['同化'],color='#17becf',lw=1.2,label='同化−实测');er.set_ylabel('相对实测的误差（m）');er.set_ylim(-.12,.16);er.set_yticks([-.1,0,.1]);er.text(.985,.1,'上限 +0.1m',transform=er.get_yaxis_transform(),ha='right',va='bottom',fontsize=8,color='#aa3333');er.text(.985,-.1,'下限 −0.1m',transform=er.get_yaxis_transform(),ha='right',va='top',fontsize=8,color='#824b9b');er.legend(loc='lower left',ncol=2,fontsize=9)
for q in [ax,er]:q.set_xlim(0,30);q.set_xticks(range(0,31,5));q.grid(color='#ddd',lw=.6)
er.annotate('青线在零线上方：同化水位偏高',xy=(11.5,errs['同化'][551]),xytext=(5,.143),fontsize=10,color='#148b98',arrowprops={'arrowstyle':'->','color':'#148b98'})
er.set_xlabel('模拟天数：2025-07-22 12:00至08-21 12:00')
fig.text(.09,.045,'本测点30天MAE估计：模型 0.91cm → 同化 4.87cm。绿色带只是±10cm参考带，不代表同化比模型更好。',fontsize=10,color='#334155');fig.savefig(O/'仅WXQ_JZZ21闸前解读.png',dpi=190);fig.savefig(O/'仅WXQ_JZZ21闸前解读.svg');plt.close(fig)
for lo,hi in [(7,14),(16,26),(27.5,30)]:
 mask=(x>=lo)&(x<=hi);print(lo,hi, '模型MAEcm',abs(errs['模型'][mask]).mean()*100,'同化MAEcm',abs(errs['同化'][mask]).mean()*100)
