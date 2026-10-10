"""同控制体完整30天残差与全线水位精度，展示不同验证目标。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
R=HERE.parents[1]
old=json.loads((R/'39_局部与全线边界归因/03_接口与水量补录/G00_冻结原参数/进度.json').read_text())
new=json.loads((HERE.parent/'06_闸流水量一致性修复/G00_更严闸流一致性/结果.json').read_text())
assert new['status']=='COMPLETED' and new['completed_cycles']==1440
scored=json.loads((HERE/'主进程精度独立重算.json').read_text())
means=[next(x['MAE'] for x in scored['rows'] if x['case']==case and x['group']=='raw118' and x['period']=='all30') for case in ['original','strict']]
residuals=[old['substep_continuity_max_abs_m3s'],new['maximum_control_volume_residual_m3s']]
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.79,wspace=.28)
labels=['原计算判据','更严闸流一致性']
colors=['#777777','#2878b5']
ax=axes[0]
ax.bar(labels,residuals,color=colors,width=.5)
ax.set_yscale('log');ax.set_ylim(1e-10,5e-5)
ax.axhline(1e-6,color='#bd4141',ls='-.',lw=1.5)
ax.text(.98,1e-6*1.3,'原门槛 1e-6 m³/s',transform=ax.get_yaxis_transform(),ha='right',fontsize=9,color='#a23737')
for i,y in enumerate(residuals):ax.text(i,y*1.6,f'{y:.3e}',ha='center',fontsize=11)
ax.set_ylabel('最大控制体水量残差（m³/s，对数轴）')
ax.set_title('兰河测流断面至闸门：全部成功子步',loc='left',fontweight='bold',fontsize=11)
ax=axes[1]
ax.bar(labels,[v*100 for v in means],color=colors,width=.5)
for i,y in enumerate(means):ax.text(i,y*100+.7,f'{y*100:.4f} cm',ha='center',fontsize=11)
ax.set_ylim(0,33);ax.set_ylabel('118个闸节点全月水位 MAE（cm）')
ax.set_title('水位平均误差仍约27.35 cm',loc='left',fontweight='bold',fontsize=11)
ax.text(.5,.87,f'MAE变化仅 +{(means[1]-means[0])*1000:.4f} mm',transform=ax.transAxes,ha='center',fontsize=10,color='#555555')
for ax in axes:
    ax.grid(axis='y',color='#e0e0e0',lw=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('数值一致性已修正，主要水位偏差仍未消除',fontsize=17,fontweight='bold',y=.965)
fig.text(.5,.075,'原输入、原初态、原n/C，无同化；仅收紧闸流一致性判据，未混入西赵河几何候选。\n新分支1440周期、172807个成功子步；7次失败按原规则回退减步。数值门槛通过不等于工程精度达标。',ha='center',fontsize=10)
fig.savefig(HERE/'数值一致性修复与水位精度.png',dpi=240)
fig.savefig(HERE/'数值一致性修复与水位精度.svg')
plt.close(fig)
