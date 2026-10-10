"""展示已确认的区间错位及候选停止位置，不拼接未完成的全月成绩。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'05_西赵河无同化验证/western_repair'
stop=json.loads((RUN/'修复区输入轮廓门槛停止.json').read_text())
point=next(p for p in stop['points'] if p['chainage_m']==69605.)
data=pd.read_csv(RUN/'未完成轮廓范围.csv')
data=data[data.chainage_m==69605.].sort_values('time_s')
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],
                     'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(2,1,figsize=(12.5,8.5),gridspec_kw={'height_ratios':[1,1.1]})
fig.subplots_adjust(left=.1,right=.975,top=.87,bottom=.135,hspace=.48)
ax=axes[0]
def segment(a,b,y,label,color):
    ax.plot([a/1000,b/1000],[y,y],color=color,lw=10,solid_capstyle='butt')
    ax.text((a+b)/2000,y+.16,label,ha='center',fontsize=9.5,color='#222222')
segment(69521,69789,2.5,'进口侧误接至出口断面：268 m','#d98236')
segment(69605,69789,2.03,'管身：184 m','#2878b5')
ax.plot([69.789,69.872],[2.5,2.5],ls='--',color='#bd4848',lw=2)
ax.text(69.8305,2.75,'出口83 m缺段',ha='center',fontsize=9,color='#a23b3b')
ax.plot([69.872,69.90],[2.5,2.5],color='#5c8a5d',lw=8)
segment(69521,69605,.94,'进口：84 m','#d98236')
segment(69605,69789,.47,'管身不变：184 m','#2878b5')
segment(69789,69872,.94,'出口：83 m','#5c8a5d')
for x in [69.521,69.605,69.789,69.872]:ax.axvline(x,color='#dddddd',lw=.7,zorder=0)
ax.set_xlim(69.505,69.905);ax.set_ylim(.15,3.0)
ax.set_xticks([69.521,69.605,69.789,69.872]);ax.set_xticklabels(['69+521','69+605','69+789','69+872'])
ax.set_yticks([2.26,.71],['原模型','修复候选']);ax.set_xlabel('工程桩号（m）')
ax.set_title('连接和区间修复有原表依据；本图仅示桩号归属，不是竣工纵剖面',loc='left',fontweight='bold',fontsize=11)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)

ax=axes[1]
ax.plot(data.time_s/86400,data.head_m,color='#2878b5',lw=1.5,label='候选进口计算水位（无同化）')
ax.plot([data.time_s.iloc[-1]/86400,point['time_s']/86400],[data.head_m.iloc[-1],point['head_m']],color='#2878b5',lw=1.5)
ax.axhline(point['input_top_m'],color='#c13f3f',ls='-.',lw=1.5,label='沿用假设的轮廓上缘 143.57 m（未核实墙顶）')
end=point['time_s']/86400
ax.scatter([end],[point['head_m']],color='#c13f3f',s=25,zorder=4)
ax.axvspan(end,30,color='#eeeeee',zorder=0)
ax.text(28,.5*(data.head_m.min()+point['input_top_m']),'停止后未运行\n不补算或拼接',ha='center',color='#666666')
ax.annotate(f'第{end:.3f}天停止\n首次超出约{point["excess_m"]*1000:.3f} mm',xy=(end,point['head_m']),xytext=(16.3,point['input_top_m']+.075),arrowprops={'arrowstyle':'->','color':'#555555'},fontsize=9)
ax.set_xlim(0,30);ax.set_ylim(data.head_m.min()-.04,point['input_top_m']+.13)
ax.set_xlabel('起算后时间（天）');ax.set_ylabel('高程（m）');ax.grid(axis='y',color='#dddddd',lw=.6)
ax.set_title('69+605进口顶部仍缺依据，候选未完成30天适用范围验证',loc='left',fontweight='bold',fontsize=11)
ax.legend(loc='upper left',fontsize=9,frameon=False);ax.spines[['top','right']].set_visible(False)
fig.suptitle('西赵河连接修复：确定的改动与尚未通过的条件',fontsize=17,fontweight='bold',y=.975)
fig.text(.5,.04,'新进口底高程、底宽和桩号来自原表交叉核对；顶部143.57 m仅沿用原轮廓高度。\n触及该假设范围不等同现场漫顶，也不能用已完成的25天结果冒充全月精度改善。',ha='center',fontsize=10)
fig.savefig(HERE/'西赵河连接修复与范围门槛.png',dpi=240)
fig.savefig(HERE/'西赵河连接修复与范围门槛.svg')
plt.close(fig)
