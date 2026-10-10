"""按冻结B02的43个测点位置，筛查实测水位与C07原始轮廓的矛盾。只读。"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
F=ROOT/'26_冻结基线与后续归因'
files={
    'operators':F/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json',
    'positions':F/'共享输入/水位测点位置审核.json',
    'model':F/'共享输入/C07.json',
    'observations':F/'共享输入/作者已核实修正观测层.csv',
}
ops=json.loads(files['operators'].read_text())
rows=json.loads(files['positions'].read_text())['sensor_rows']
base=json.loads(files['model'].read_text())['中线全线']['baseData']
observations={}
with files['observations'].open() as f:
    for r in csv.DictReader(f):
        if r['valid'].lower()=='true' and 0<float(r['time_s'])<=2592000:
            observations.setdefault((r['station_code'],r['metric']),[]).append(r)
results=[]
for o in ops:
    side={'forebay_m':'front','afterbay_m':'back'}[o['metric']]
    positions=[r for r in rows if r['stationcode']==o['station'] and r['metric']==side and r.get('sensor_chainage')==o['chainage']]
    brackets={json.dumps(b,sort_keys=True) for r in positions for b in r['model_brackets']}
    assert len(brackets)==1,(o,brackets)
    b=json.loads(next(iter(brackets)))
    channel=base['channels'][b['channel']]
    la,lb=b['left_section'],b['right_section']
    assert la in channel['sec_names'] and lb in channel['sec_names']
    sa,sb=base['sections'][la],base['sections'][lb]
    assert sa['location']<=o['chainage']<=sb['location']
    f=(o['chainage']-sa['location'])/(sb['location']-sa['location'])
    edges=[(1-f)*sa['yz'][k][1]+f*sb['yz'][k][1] for k in [0,-1]]
    bottom=(1-f)*min(z for _,z in sa['yz'])+f*min(z for _,z in sb['yz'])
    assert abs(bottom-o['offset'])<1e-9,(o,bottom)
    rr=sorted(observations[(o['station'],o['metric'])],key=lambda x:float(x['time_s']))
    assert len(rr)==1440 and len({r['time_s'] for r in rr})==1440
    h=np.array([float(r['value']) for r in rr]);raw=np.array([float(r['raw_value']) for r in rr])
    excess=h-min(edges); mask=excess>0.1
    nmax=cur=0
    for x in mask:
        cur=cur+1 if x else 0;nmax=max(nmax,cur)
    results.append(dict(station=o['station'],station_name=positions[0]['station_name'],metric=o['metric'],assimilated=o['assimilated'],
        chainage_m=o['chainage'],channel=b['channel'],left_section=la,right_section=lb,right_weight=f,
        left_edge_m=edges[0],right_edge_m=edges[1],bottom_m=bottom,source_position_evidence=sorted({r['evidence'] for r in positions}),
        historical_sensor_identity=sorted({r['historical_sensor_identity'] for r in positions}),
        count=len(rr),observed_min_m=float(h.min()),observed_max_m=float(h.max()),observed_mean_m=float(h.mean()),
        exceed_min_m=float(excess.min()),exceed_max_m=float(excess.max()),exceed_mean_m=float(excess.mean()),
        above_0_1m_count=int(mask.sum()),above_0_1m_fraction=float(mask.mean()),max_consecutive_samples=nmax,
        max_consecutive_covered_hours=max(nmax-1,0)*.5,
        raw_above_0_1m_count=int((raw-min(edges)>.1).sum()),author_offset_m=sorted({float(r['author_offset_m']) for r in rr}),
        first_exceed_timestamp=next((r['timestamp'] for r,m in zip(rr,mask) if m),None),
        last_exceed_timestamp=next((r['timestamp'] for r,m in reversed(list(zip(rr,mask))) if m),None)))
results.sort(key=lambda r:r['exceed_max_m'],reverse=True)
out={'sources':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()}for k,v in files.items()},
    'method':'43 frozen operators; 0<t<=2592000; interpolate original C07 left/right edge elevations along chainage; threshold observation - min(two edges)>0.1m; no simulation',
    'point_count':len(results),'above_0_1m_point_count':sum(r['above_0_1m_count']>0 for r in results),'results':results,
    'restriction':'安装资料与2025历史列仪器身份和统一高程基准仍未完全闭合；输入边缘高程未证实是工程堤顶，不能解释成现场漫顶。'}
(OUT/'实测与断面轮廓核查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
hit=[r for r in results if r['above_0_1m_count']>0]
text=['# 实测水位与输入断面轮廓的条件性核查','',
      '**43个有冻结位置算子的水位测点中，13个的实测水位曾高出对应输入轮廓顶0.1m以上；5个在1440个半小时时刻全部高出。JZZ15闸后尤为明确：即使完全不读模型计算水位，实测与输入轮廓的矛盾仍存在。**','',
      '本次只使用冻结B02的43个算子、共享C07原始yz轮廓、冻结位置审核及冻结观测，不进行水力仿真。按安装桩号在原始上下游断面之间线性插值两端边缘高程，再取两个边缘中的较低值。43个测点的两端边缘实际相等，插值渠底与冻结算子的offset逐个一致（差<10⁻⁹m）。统计时段为2025年7月22日12:30至8月21日12:00，排除起点，逐点1440个有效值。','',
      '## 超过0.1m的测点','',
      '|水位测点|用途|输入轮廓顶(m)|实测范围(m)|最大超出(m)|超过0.1m次数／比例|',
      '|---|---|---:|---:|---:|---:|']
for r in hit:
    side='闸前' if r['metric']=='forebay_m' else '闸后'
    use='参与同化' if r['assimilated'] else '检验'
    text.append(f"|{r['station']} {r['station_name']}{side}|{use}|{r['left_edge_m']:.3f}|{r['observed_min_m']:.3f}–{r['observed_max_m']:.3f}|{r['exceed_max_m']:.3f}|{r['above_0_1m_count']}/1440，{100*r['above_0_1m_fraction']:.2f}%|")
text+=['','其余30个没有出现超过0.1m的样本。这13个中11个参与同化、2个用于检验；5个整月全部超过的测点是JZZ15、JZZ37、JZZ47、JZZ50、JZZ58闸后。上述比例为半小时样本比例，不推定采样间隔内全过程连续超出。','',
       '## JZZ15闸后的查源链','',
       '- 位置材料记录为北汝河倒虹吸出口节制闸闸后左、右岸水位计，均为K280+683，“弧形闸门闸后下游约100米”；冻结审核来源为“附件 Sheet1 B160:F160、B161:F161”。两支仪器的桩号相同，核查没有将它们的读数另行平均。',
       '- 冻结算子对应POOL15-1-JJZ15DHX20，位于原断面q15s3（280583m）与q15s4（285111m）之间，距前者100m，右断面权重0.02208480565。',
       '- 建模Excel“渠道”!C153:J153给出CH152入口顶126.877m、出口顶126.730m；插值顶高程126.873754m、底高程118.443754m、高度8.430m。',
       '- 实测最小127.769166m，最大128.398207m，整月均高于轮廓顶0.895412–1.524453m，平均高出1.149343m；所有1440个样本均超过0.1m。',
       '- 本点author_offset_m=0，raw_value复核仍为1440/1440次超出，排除了“作者偏移修正把实测整体抬高”的解释。其余12个命中测点也均没有作者偏移。','',
       '## 如何解释','',
       '**至少在这些测点，超出轮廓不能全部解释为模型单方面把水位算高。** 在当前冻结的安装位置和高程对应假设下，实测水位本身也超过所输入的几何范围。优先要核实原断面、轮廓顶高程语义、真实结构类别，以及仪器与历史数据列的对应关系。','',
       '这不证明实际漫顶，也不证明实测一定对或原断面一定错。冻结位置审核明确记录历史仪器身份“未给定”，未闭合2025年历史水位列到底来自哪支仪器；高程基准尚需独立核实；安装桩号与历史工况是否一致也未得到新增资料确认。“顶高程”也不能在未查图纸前直接认定为堤顶或墙顶。不得据此自动抬高几何、调整观测偏置或改变留出测点。','',
       '逐测点证据及源文件SHA256见`实测与断面轮廓核查.json`；复核脚本为`实测与断面轮廓核查.py`。原报告主文及全部冻结资料未修改。']
(OUT/'实测与断面轮廓条件性核查.md').write_text('\n'.join(text)+'\n')
print(json.dumps({k:out[k]for k in ['point_count','above_0_1m_point_count']}))
print(json.dumps([r for r in results if r['above_0_1m_count']>0],ensure_ascii=False,indent=2))
