"""澎河同状态端面面积对照；不是两种实测断面或水动力过程结果。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
evidence = json.loads((HERE/'主进程澎河端面独立核验.json').read_text())
row = next(x for x in evidence['same_state_rows'] if x['state'] == '原参数无同化30天末态')
depth = row['mean_depth_m']
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],
                    'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
fig.subplots_adjust(left=.07, right=.98, bottom=.27, top=.73, wspace=.16)
for ax, prefix, title, color in zip(axes, ['old','endpoint'],
        ['原端面取值：邻接计算段中点参数','修正端面取值：q11s25 实际端断面'],
        ['#777777','#2878b5']):
    height, width, slope = row['old_g1_g2_g3'] if prefix == 'old' else row['endpoint_g1_g2_g3']
    xt = width/2 + slope*height
    xw = width/2 + slope*depth
    ax.plot([-xt,-width/2,width/2,xt],[height,0,0,height],color=color,lw=2.2)
    ax.fill([-xw,-width/2,width/2,xw],[depth,0,0,depth],color='#74bddb',alpha=.35)
    ax.plot([-xw,xw],[depth,depth],color='#257daa',lw=1.4)
    ax.text(0,depth+.5,f'同一平均水深 {depth:.4f} m',ha='center',fontsize=10)
    ax.text(0,2.2,f"A = {row[prefix+'_geometry']['A']:.3f} m²",ha='center',fontsize=14,color=color,fontweight='bold')
    ax.text(0,.65,f'底宽 {width:g} m；边坡 {slope:g}',ha='center',fontsize=10)
    ax.set_xlim(-23,23);ax.set_ylim(-.4,9.6)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
    ax.set_xlabel('横向位置（m；两图横纵比例一致）')
    ax.grid(axis='y',color='#dddddd',lw=.6)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.5,-.29,f"原 C 与 Q 不变，端头项 {row[prefix+'_end_head_term_magnitude_m']*100:.3f} cm",
            transform=ax.transAxes,ha='center',fontsize=11,color=color)
axes[0].set_ylabel('相对同一端部渠底的高度（m）')
fig.suptitle('端部床高取端断面，过水面积却曾取邻接段中点',fontsize=17,fontweight='bold',y=.965)
fig.text(.5,.845,'澎河渡槽进口 K232+036；原参数30天末保存状态的纯代数对照',ha='center',fontsize=11)
fig.text(.5,.055,'原表渠道行114与渐变段行95支持端断面底高程127 m、底宽22 m、边坡0、顶高程135 m。\n这里只比较同状态下的取值；水头项变化不等于水位过程或精度收益，未改糙率、闸流系数与端损系数。',
         ha='center',fontsize=10)
fig.savefig(HERE/'澎河端面几何取值差异.png',dpi=240)
fig.savefig(HERE/'澎河端面几何取值差异.svg')
plt.close(fig)
