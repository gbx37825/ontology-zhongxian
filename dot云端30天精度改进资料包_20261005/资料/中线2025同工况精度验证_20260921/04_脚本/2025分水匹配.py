"""Audit raw 2025 offtake observations and match existing 128 bleeders, without forcing fill."""
import json,re
from pathlib import Path
import pandas as pd,numpy as np,openpyxl
R=Path(__file__).resolve().parents[1];D=R/'02_输入';d=pd.read_csv(D/'2025分退水原始窗口.csv');d['时间']=pd.to_datetime(d['时间']);formal_start=pd.Timestamp('2025-07-22 12:00');end=pd.Timestamp('2025-08-21 12:00');grid=pd.date_range(formal_start,end,freq='30min')
old=json.loads((D/'B04_完整时序来源映射.json').read_text());old={x['target']:x for x in old if x['type']=='bleeders'};model=next(iter(json.loads((D/'B04_分水身份修正30天.json').read_text()).values()))['baseData']
previous=json.loads((R/'07_对比分析/00时窗口旧审核归档/2025分退水质量审核.json').read_text())
installation=list({(q['row'],q['name']):q for st in previous['all_source_quality'] for q in st['installation']}.values())
def norm(s):return str(s or '').replace('（','(').replace('）',')').replace('望成岗','望城岗').replace('溃城寨','聩城寨').replace('荆柯','荆轲')
def installmatch(name):
 return [{**x,'evidence':f'安装补充表 Sheet1 B{x["row"]}:G{x["row"]}'} for x in installation if norm(x['name'])==norm(name) or (name=='民有渠分水口' and x['name']=='民有渠临时分水闸') or (name in ['上庄分水口','新增上庄分水口'] and '上庄分水口' in x['name'])]
def longest_gap(valid):
 missing=~valid;best=cur=0
 for z in missing:cur=cur+1 if z else 0;best=max(best,cur)
 return best
names=sorted(d['闸站名称'].unique());stats=[];clean=[]
for name,g in d.groupby('闸站名称',sort=True):
 g=g.sort_values('时间');formal=g[g['时间'].between(formal_start,end)].copy();holes=g[['孔1瞬时流量','孔2瞬时流量','孔3瞬时流量']].sum(axis=1,min_count=1);delta=g['总瞬时流量']-holes;invalid=formal['总瞬时流量'].isna()|(formal['总瞬时流量']<=-100);values=formal['总瞬时流量'].mask(invalid);lookup=pd.Series(values.to_numpy(),index=formal['时间']);aligned=lookup.reindex(grid) if not lookup.index.duplicated().any() else pd.Series(index=grid,dtype=float)
 st={'name':name,'codes':[str(x) for x in g['闸站编码'].unique()],'raw_n':len(g),'formal_n':len(formal),'duplicate_times':int(g['时间'].duplicated().sum()),'off_grid':int((((formal['时间']-formal_start).dt.total_seconds())%1800!=0).sum()),'formal_expected_n':len(grid),'formal_source_row_first':int(formal['源行号'].min()) if len(formal) else None,'formal_source_row_last':int(formal['源行号'].max()) if len(formal) else None,'formal_missing_or_invalid_rate':float((len(grid)-aligned.notna().sum())/len(grid)),'formal_missing_rows':int(len(grid.difference(formal['时间']))),'formal_invalid_sentinel':int(invalid.sum()),'formal_valid_n':int(values.notna().sum()),'formal_negative_nonsentinel':int(((values<0)&values.notna()).sum()),'formal_min_valid':float(values.min()) if values.notna().any() else None,'formal_max_valid':float(values.max()) if values.notna().any() else None,'formal_mean_valid':float(values.mean()) if values.notna().any() else None,'formal_nonzero_n':int((values.abs()>1e-8).sum()),'longest_missing_or_invalid_samples':longest_gap(aligned.notna().to_numpy()),'total_minus_holes_max_abs':float(delta.abs().max()),'source_row_first':int(g['源行号'].min()),'source_row_last':int(g['源行号'].max()),'installation':installmatch(name)};stats.append(st)
 for x in formal.to_dict('records'):
  invalid=pd.isna(x['总瞬时流量']) or x['总瞬时流量']<=-100
  clean.append({'source_name':name,'source_code':x['闸站编码'],'source_row':x['源行号'],'timestamp':str(x['时间']),'total_raw':x['总瞬时流量'],'flow_validated':None if invalid else x['总瞬时流量'],'valid':not invalid,'flag':'明确负哨兵或缺值' if invalid else '负流量待核方向' if x['总瞬时流量']<0 else '有效原总量'})
byname={x['name']:x for x in stats};mapping=[];used=set();candidate=set()
for target,b in model['bleeders'].items():
 o=old[target];cols=o.get('columns') or ([o.get('column')] if o.get('column') else []);note='B04既有身份映射与2025源站名一致'
 if target.startswith('FSK1-'):cols=['肖楼分水口'];note='模型4196与安装补充表D3=4196一致，2025新增真记录'
 if target.startswith('FSK2-'):cols=['刁河退水闸'];note='模型14538与安装补充表D10=14537.9一致，2025全零为真记录'
 selected=[];state='已匹配'
 for col in cols:
  hits=[n for n in names if norm(n)==norm(col)]
  if len(hits)==1:selected.append(hits[0])
  else:state='缺直接同名源站'
 if target.startswith('FSK121-'):
  selected=['西黑山进口闸'];state='候选待确认';note='旧名西黑山引水闸；2025为西黑山进口闸，安装表B749:G751为XW支线0公里进口；主体模型1121720桩号尚无直接换算证明，不自动当已确认绑定';candidate.update(selected)
 else:used.update(selected)
 evidence=[]
 for n in selected:
  st=byname[n];evidence.append(f'分水口.xlsx/20250501-20250831 A:G 源行{st["source_row_first"]}—{st["source_row_last"]}中的{n}；逐点行号见原始窗口')
  evidence += [q['evidence'] for q in st['installation']]
 mapping.append({'target':target,'model_channel':b['channel'][0],'model_chainage':b['channel'][1],'old_columns':cols,'source_names':selected,'source_codes':[byname[n]['codes'] for n in selected],'status':state,'aggregation':'同时间各源站总瞬时流量相加，各源必需有效；绝不总列再加孔列' if len(selected)>1 else '仅总瞬时流量','note':note,'evidence':evidence,'formal_source_quality':[{k:v for k,v in byname[n].items() if k!='installation'} for n in selected]})
unbound=[{**s,'candidate_target':'FSK121-2-pool56' if s['name'] in candidate else None,'scope_note':'位置证据若为XW支线不得再次作为干渠独立分水叠加；其余无桩号时不猜位置'} for s in stats if s['name'] not in used]
for u in unbound:
 if u['name']=='西黑山进口闸':
  u['scope_class']='XW支线入口/FSK121候选'
  u['scope_evidence']='安装表749—751明确XW0+087.5/207.5/399.5；与旧FSK121别名待主输入假设确认。'
 elif u['name']=='文村北调节池':
  u['scope_class']='XW支线下游串联断面，不另加干渠取水'
  u['scope_evidence']='安装表738—743明确XW10+526.1/746，与西黑山进口XW0公里同一里程系统，下游约10km；直接再扣会把入口已计的支线水量重复计入。'
 elif any(z in u['name'] for z in ['王庆坨','杨柳青','子牙河北','外环河']):
  u['scope_class']='天津方向或支线内部疑似，未证实为主渠新分水'
  u['scope_evidence']='仅源站名称含天津方向地名/连接井上下游/出口、分流井功能；材料无可映射主线桩号，不据地名认定精确拓扑，不强入干渠。'
 else:
  u['scope_class']='位置未闭合，不能确认主线新增'
  u['scope_evidence']='当前安装表无同名明确干渠桩号；闸站编码不等于沿程桩号，不能按编码顺序归渠池。'
unbound.sort(key=lambda x: -(x['formal_mean_valid'] or 0))
summary={'window':'2025-07-20 00:00..2025-08-21 12:00','formal_window':'2025-07-22 12:00..2025-08-21 12:00 inclusive','formal_grid_points':len(grid),'source_stations':len(names),'raw_rows':len(d),'matched_targets':sum(x['status']=='已匹配' for x in mapping),'candidate_targets':[x['target'] for x in mapping if x['status']!='已匹配'],'unbound_source_stations':len(unbound),'total_holes_consistency':'全部总量与现存各孔求和差<=0.00500001，为0.01舍入一致。只用总量，不重复叠加。缺孔可能为未设孔，不能据此认定缺测，也不补孔。','unit':'表头未注明单位；按瞬时流量数值及旧模型m3/s作待确认单位假设，未自动缩放。','invalid_rule':'总量<=-100为明确负哨兵隔离；其他负数保留但标记，不默默清零；不插值、不末值延用。','all_source_quality':stats,'mapping':mapping,'unbound_sources':unbound}
assert len(mapping)==128;assert all(x['total_minus_holes_max_abs']<=.00500001 for x in stats)
(D/'2025分退水映射.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
def save(rows,path):pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in x.items()} for x in rows]).to_csv(path,index=False)
save(mapping,D/'2025分退水映射.csv');save(clean,D/'2025分退水有效观测.csv');save(stats,R/'07_对比分析/2025分退水逐站质量.csv');save(unbound,R/'07_对比分析/2025未绑定分退水源站.csv');(R/'07_对比分析/2025分退水质量审核.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# 2025分退水提取与模型匹配','',f'原始窗口{len(names)}站{len(d)}行，目标30天每站应1441个半小时点。128个模型分退水口中127个明确匹配，FSK121西黑山引水闸暂保留西黑山进口闸候选，不自动确认。','',summary['total_holes_consistency'],summary['unit'],summary['invalid_rule'],'', '## 关键身份','', 'FSK1肖楼、FSK2刁河已有真观测；FSK101仍为上庄与新增上庄同时间总量相加，FSK102南新城、FSK103田庄、FSK73民有渠。FSK46原模型与2025同名黄河退水闸可直接取值；这不另外证明安装表穿黄退水闸的里程别名。','', '## 正式窗口坏点或缺测','', '|站名|缺行|无效哨兵|其他负值|最长连续缺/坏点数|','|---|---:|---:|---:|---:|']
for x in stats:
 if x['formal_missing_rows'] or x['formal_invalid_sentinel'] or x['formal_negative_nonsentinel']:lines.append(f'|{x["name"]}|{x["formal_missing_rows"]}|{x["formal_invalid_sentinel"]}|{x["formal_negative_nonsentinel"]}|{x["longest_missing_or_invalid_samples"]}|')
lines+=['','## 未绑定源站（不能把支线重复计为干渠分水）','', '|站名|有效平均流量*|最大流量*|有效非零点|位置证据|','|---|---:|---:|---:|---|']
for x in unbound:lines.append(f'|{x["name"]}|{x["formal_mean_valid"]}|{x["formal_max_valid"]}|{x["formal_nonzero_n"]}|'+ '; '.join(f'{z["chainage_raw"]} / {z["evidence"]}' for z in x['installation'])+'|')
lines+=['','*单位暂按m³/s，缺失及明确坏值不参与均值；这些流量不可跨站简单相加。西黑山进口在安装表749—751的XW0公里，文村北在738—743的XW10公里，属于同里程系统的入口与下游断面，文村北不能再作为干渠独立出口扣除。王庆坨、杨柳青、子牙河北、外环河仅根据地名/功能列为天津方向或支线内部疑似，未据此硬认拓扑。其余非零未绑站也缺主线桩号，目前没有一处可以确认是主线新增节点。','', '30min网格复核改用时间差.total_seconds，全部源站formal off_grid=0；此前datetime64[us]整数被按ns解释造成误报，不是实际时间异常。正式窗口包含08-21 12:00，分母统一1441；原始source_row范围含07-20起两天先导，新增formal_source_row_first/last专列正式段。未改模型、未补空值、未运行仿真。逐点原始Excel行号已保留。']
(R/'08_报告/2025分退水数据与映射审核.md').write_text('\n'.join(lines))
print({k:v for k,v in summary.items() if k not in ['all_source_quality','mapping','unbound_sources']});print('GAPS',[(s['name'],s['formal_invalid_sentinel'],s['longest_missing_or_invalid_samples']) for s in stats if s['formal_invalid_sentinel']]);print('UNBOUND',[(s['name'],s['formal_mean_valid']) for s in unbound])
