from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
model=pd.read_csv(O/'M00_原输入完整30天/半小时水位流量.csv').melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim')
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');obs=obs[obs.valid].rename(columns={'station_code':'station','value':'observed'});merged=model.merge(obs,on=['station','time_s','metric'],validate='one_to_one');merged['error']=merged.sim-merged.observed;top=merged.loc[merged.error.abs().nlargest(10).index].copy();names=pd.read_csv(R/'02_输入/2025节制闸站名映射.csv').set_index('station_code');raw=pd.read_csv(R/'02_输入/2025节制闸原始窗口.csv',parse_dates=['时间']);records=[];neighbors=[];request_rows=set();roles={}
for bs in ['B01_允许修改水位的状态同化参考基线','B02_渠池归属修复后的参数同化诊断基线']:
 roles[bs]=json.loads((B/bs/'同化运行/观测算子与留出.json').read_text())
indexed=merged.set_index(['station','metric','time_s']);rawindex=raw.set_index(['站点名称','时间'])
for rank,r in enumerate(top.itertuples(),1):
 station=r.station;col='闸前水位' if r.metric=='forebay_m' else '闸后水位';name=names.loc[station,'source_name'];rr=rawindex.loc[(name,pd.Timestamp(r.timestamp))];prev=indexed.loc[(station,r.metric,r.time_s-1800)];nxt=indexed.loc[(station,r.metric,r.time_s+1800)]
 rs={}
 for bs,ops in roles.items():
  op=[x for x in ops if x['station']==station and x['metric']==r.metric];rs[bs[:3]]='参与同化测点' if op and op[0]['assimilated'] else '9个留出检验测点之一' if op else '全线评价点，未进入43个明确位置测点集合（不是9个留出点）'
 records.append(dict(rank=rank,station=station,name=name,metric=r.metric,timestamp=r.timestamp,time_s=r.time_s,model=r.sim,observed=r.observed,error=r.error,abs_error=abs(r.error),raw_source_row=int(rr['源行号']),raw_long_table_value=float(rr[col]),frozen_raw_value=float(r.raw_value),author_offset_m=float(r.author_offset_m),valid=bool(r.valid),pdf_comparable=bool(r.pdf_comparable),previous_observed=float(prev.observed),next_observed=float(nxt.observed),jump_from_previous_m=float(r.observed-prev.observed),jump_to_next_m=float(nxt.observed-r.observed),previous_next_difference_m=float(nxt.observed-prev.observed),previous_model=float(prev.sim),next_model=float(nxt.sim),B01_role=rs['B01'],B02_role=rs['B02']))
 for tt in [pd.Timestamp(r.timestamp)+pd.Timedelta(minutes=i) for i in [-60,-30,0,30,60]]:
  row=rawindex.loc[(name,tt)];request_rows.add(int(row['源行号']));neighbors.append(dict(focus_rank=rank,focus_timestamp=r.timestamp,station=station,timestamp=str(tt),source_row=int(row['源行号']),**{k:(None if pd.isna(row[k]) else float(row[k])) for k in ['闸前水位','闸后水位','瞬时流量','孔1实际开度','孔2实际开度','孔3实际开度','孔4实际开度']}))
pd.DataFrame(records).to_csv(O/'M00最大十个误差及相邻变化.csv',index=False);pd.DataFrame(neighbors).to_csv(O/'最大误差前后原始观测.csv',index=False)
# 回查源Excel XML实际行，不依赖已经提取的长表数值。保留不明单位，不改任何单元格。
source=R/'01_原始资料/节制闸.xlsx';ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};xmlrows=[]
with zipfile.ZipFile(source) as z:
 strings=[]
 if 'xl/sharedStrings.xml' in z.namelist():
  for _,e in ET.iterparse(z.open('xl/sharedStrings.xml'),events=('end',)):
   if e.tag.endswith('}si'):strings.append(''.join(e.itertext()));e.clear()
 wb=ET.fromstring(z.read('xl/workbook.xml'));rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));targets={e.attrib['Id']:e.attrib['Target'] for e in rels};sheet=next(e for e in wb.find('m:sheets',ns) if e.attrib['name']=='节制闸');target=targets[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']];xmlname=target.lstrip('/') if target.startswith('/') else 'xl/'+target;tag='{'+ns['m']+'}';limit=max(request_rows)
 for _,e in ET.iterparse(z.open(xmlname),events=('end',)):
  if e.tag!=tag+'row':continue
  rownum=int(e.attrib['r'])
  if rownum in request_rows:
   values={}
   for cell in e:
    value=cell.find(tag+'v');txt=None if value is None else value.text
    if cell.attrib.get('t')=='s':txt=strings[int(txt)]
    elif cell.attrib.get('t')=='inlineStr':txt=''.join(cell.itertext())
    values[cell.attrib['r']]=txt
   xmlrows.append(dict(source_row=rownum,cells=values))
  e.clear()
  if rownum>=limit:break
byrow={x['source_row']:x['cells'] for x in xmlrows};checks=[]
for rec in records:
 row=rec['raw_source_row'];cells=byrow[row];field='D' if rec['metric']=='forebay_m' else 'E';value=float(cells[field+str(row)]);checks.append(dict(rank=rec['rank'],source_row=row,cell=field+str(row),excel_value=value,raw_long_table_value=rec['raw_long_table_value'],frozen_raw_value=rec['frozen_raw_value'],frozen_observed=rec['observed'],match=abs(value-rec['raw_long_table_value'])<1e-12 and abs(value-rec['frozen_raw_value'])<1e-12 and abs(value+rec['author_offset_m']-rec['observed'])<1e-12));assert checks[-1]['match']
q=merged[(merged.station=='JZZ26')&(merged.metric=='forebay_m')];monthly=dict(station='JZZ26',metric='forebay_m',MAE=float(q.error.abs().mean()),bias=float(q.error.mean()),negative_errors=int((q.error<0).sum()),samples=len(q),model_min=float(q.sim.min()),model_max=float(q.sim.max()),observed_min=float(q.observed.min()),observed_max=float(q.observed.max()),max_halfhour_observation_jump=float(q.sort_values('time_s').observed.diff().abs().max()))
update=[]
for bs in roles:
 log=pd.read_csv(B/bs/'同化运行/逐观测更新记录.csv');r=log[(log.time_s==2068200)&(log.station=='JZZ49')].iloc[0];update.append(dict(baseline=bs,station='JZZ49',metric='afterbay_m',time_s=2068200,status=r.status,observed=float(r.observed),forecast_at_mapped_sensor=float(r.forecast),innovation=float(r.innovation)))
result=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sheet='节制闸',excel_rows=xmlrows,ten_records_exact_raw_excel_match=checks,JZZ26_month=monthly,JZZ49_spike_assimilation_status=update,notes='只读核查，未剔除或替换观测。最大十记录全部沿用既定有效层；M00自身没有同化。')
(O/'最大误差原表核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print(pd.DataFrame(records).to_string(index=False));print(json.dumps(monthly,ensure_ascii=False));print(update)
