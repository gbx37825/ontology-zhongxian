from pathlib import Path
import pandas as pd,numpy as np,json
P=Path(__file__).resolve().parents[1];B=P.parent/'23_五十成员参数同化'
old=pd.read_csv(B/'30天参数同化精度汇总.csv');new=pd.read_csv(P/'30天参数同化精度汇总.csv');rows=[]
for scope in new.scope.unique():
 oa=old[(old.scope==scope)&old.run.str.startswith('P01')].iloc[0];ob=old[(old.scope==scope)&old.run.str.startswith('P02')].iloc[0];na=new[(new.scope==scope)&new.run.str.startswith('F03')].iloc[0];nb=new[(new.scope==scope)&new.run.str.startswith('F04')].iloc[0]
 for metric in ['mae','rmse','max_error']:
  rows.append(dict(scope=scope,metric=metric,old_assim=oa[metric],old_control=ob[metric],new_assim=na[metric],new_control=nb[metric],old_effect=oa[metric]-ob[metric],new_effect=na[metric]-nb[metric],difference_of_effects=(na[metric]-nb[metric])-(oa[metric]-ob[metric])))
diff=pd.DataFrame(rows);diff.to_csv(P/'拓扑修复四组差分比较.csv',index=False)
a=pd.read_csv(B/'30天参数同化逐侧精度.csv');b=pd.read_csv(P/'30天参数同化逐侧精度.csv');keys=['operator','station','metric'];parts=[a[a.run.str.startswith('P01')],a[a.run.str.startswith('P02')],b[b.run.str.startswith('F03')],b[b.run.str.startswith('F04')]]
tables=[]
for frame,label in zip(parts,['old_assim','old_control','new_assim','new_control']):tables.append(frame.set_index(keys)[['mae','rmse','max_error']].add_prefix(label+'_'))
x=pd.concat(tables,axis=1).reset_index();x['difference_of_effects_mae']=(x.new_assim_mae-x.new_control_mae)-(x.old_assim_mae-x.old_control_mae);x.to_csv(P/'拓扑修复逐侧差分.csv',index=False)
focus=x[x.station.isin(['JZZ3','JZZ4','JZZ13','JZZ14','JZZ15','JZZ35','JZZ45','JZZ55'])];focus.to_csv(P/'重点渠段四组对照.csv',index=False)
checks=json.loads((P/'30天参数同化完成核验.json').read_text());oldchecks=json.loads((B/'30天参数同化完成核验.json').read_text());tr=pd.read_csv(P/'30天参数轨迹与触界汇总.csv');otr=pd.read_csv(B/'30天参数轨迹与触界汇总.csv');param=tr[tr.run.str.startswith('F03')].merge(otr[otr.run.str.startswith('P01')],on='parameter',suffixes=('_new','_old'));param.to_csv(P/'新旧参数轨迹比较.csv',index=False)
lines=['# 渠池拓扑归属修复30天归因报告','','## 结论','','渠池归属错误已修复，但本组30天试验没有获得可靠精度收益，不能替换基线。修复后留出9侧MAE为0.248780m，高于匹配无同化的0.157748m（恶化约57.7%），也高于旧参数同化的0.184442m。作者88侧MAE从旧同化0.220636m升至0.238326m。全线MAE从0.251432m略降到0.250171m，但RMSE上升，不能用这一点平均改善掩盖留出恶化。','','因此必须收窄此前的归因：该错误确实会把异地糙率绑定并绕过局地化约束，但不是“修好它就能消除本轮精度退化”的已证实主因。旧错误路径可能包含偶然的误差补偿，现有试验不能唯一证明该解释，更不能据此恢复错误分组。','','## 完整性与同化约束','','F03和F04均完成1440周期，覆盖2025-07-22 12:00至2025-08-21 12:00。每侧每阶段1440个评分时刻，首个1800s、末个2592000s，无重复或缺时；每组463680条评分。50成员的六类水力数组和119×50参数均有限，末态与最新检查点逐元素一致。观测值已回查既定修正观测层。每周期state_unchanged均为真，即时蓄水调整全部0，同一时刻参数更新前后水位严格一致；各4个工作进程的Numba实际执行证据通过。','','## 四组同口径比较','','下表全部是消费当前观测前的半小时先验。误差单位m。效应=同化误差−本组无同化误差；差分=修复效应−旧效应，正值表示修复后同化收益变差。最大误差的差分仅为指标对照，不代表同一个极值事件的物理分解。','','|范围|指标|旧无同化|旧同化|修复无同化|修复同化|效应差分|','|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['scope']}|{r['metric']}|{r['old_control']:.6f}|{r['old_assim']:.6f}|{r['new_control']:.6f}|{r['new_assim']:.6f}|{r['difference_of_effects']:+.6f}|")
lines+=['','新旧无同化总体MAE变化只有微米到百分之一毫米量级，说明本组宏观退化不宜解释成初始参数映射本身造成的无同化水力差异；显著差异发生在反复参数更新过程中。但这仍只是一组种子、一个月份，不能当作跨工况普遍结论。','','## 重点渠段逐侧结果','','下表按闸端口径；同安装位置的完整明细另见重点渠段四组对照.csv。','','|测点|旧无同化MAE|旧同化MAE|修复无同化MAE|修复同化MAE|效应差分|','|---|---:|---:|---:|---:|---:|']
for r in focus[focus.operator=='闸端全线'].itertuples():lines.append(f'|{r.station} {r.metric}|{r.old_control_mae:.6f}|{r.old_assim_mae:.6f}|{r.new_control_mae:.6f}|{r.new_assim_mae:.6f}|{r.difference_of_effects_mae:+.6f}|')
lines+=['','## 参数轨迹和拒收','','|项目|旧参数同化|修复参数同化|','|---|---:|---:|',f"|接受观测|{oldchecks[0]['accepted']}|{checks[0]['accepted']}|",f"|大创新拒收|{oldchecks[0]['rejected']}|{checks[0]['rejected']}|",f"|离散度补充参数周期|{oldchecks[0]['rejuvenated_parameter_cycles']}|{checks[0]['rejuvenated_parameter_cycles']}|",'','大创新拒收由2658次增至6494次，约占48960次候选观测的13.3%。拒收原因可能是模型偏差、算子问题或观测异常，不能直接当作删除坏数据的依据。逐站拒收统计见大创新拒收逐测点统计.csv。','','|参数|旧末态平均倍率|修复末态平均倍率|修复期最小倍率|修复期最大倍率|','|---|---:|---:|---:|---:|']
for r in param[param.parameter.isin(['POOL3_n','POOL13_n','POOL14_n','POOL34_n','POOL35_n','POOL44_n','POOL45_n','POOL54_n','POOL55_n','JZZ13_C','JZZ14_C','JZZ35_C','JZZ45_C','JZZ55_C'])].itertuples():lines.append(f'|{r.parameter}|{r.final_mean_old:.6f}|{r.final_mean_new:.6f}|{r.minimum_new:.6f}|{r.maximum_new:.6f}|')
for n in new.run.unique():
 t=tr[tr.run==n];lines.append(f"\n{n}：轨迹未越配置参数边界；下界触及{int(t.lower_member_cycles.sum())}、上界触及{int(t.upper_member_cycles.sum())}个成员×参数×周期；末步处于边界{int((t.final_lower+t.final_upper).sum())}个成员×参数。无同化触界主要是初始集合裁剪，不是在线同化导致。")
lines+=['','## 能确认和仍不能确认的根因','','- 能确认：误命名渠道原来被并入异地渠池，造成参数错误共享与230km零距离区间；拓扑修复消除了这条错误更新路径。','- 能确认：本次没有改动31个参数无更新观测的覆盖空档，也没有解决穿黄近闸测点、剩余局损/断面、流量口径和C07长闭闸假设。','- 不能确认：留出退化唯一由某一个糙率或闸流系数造成。需要从相同检查点分支，冻结邻段参数、分别只更新n或C，核对响应方向和误差传播。','- 随机更新还有一个复现实验边界：接受观测集合变化会改变随机扰动抽样的矩阵维度，后续随机流可能分岔。因此远区变化不能直接解读为某条渠道具有同等强度的远距离物理作用；后续应冻结扰动样本或做重复种子检验。','','## 运行成本及交付边界','']
for q in checks:
 st=q['resumed_segment_status'];lines.append(f"- {q['run']}：从6小时检查点续算耗时{st['wall_seconds']:.1f}s，减步重试{st['retries']}次；不是完整从零耗时，也不是改变收敛阈值。")
lines+=['','所有比较均使用已知历史边界与闸门开度的回放，不是未知未来输入下的独立预报，不证明6小时自由预报达标。原始数据与旧结果均保留，本轮未部署平台。监测在报告完成后结束，不归档任务。']
(P/'拓扑修复30天归因报告.md').write_text('\n'.join(lines)+'\n');print(diff[diff.metric=='mae'].to_string(index=False));print(focus[focus.operator=='闸端全线'][['station','metric','old_assim_mae','new_assim_mae','difference_of_effects_mae']].to_string(index=False))
