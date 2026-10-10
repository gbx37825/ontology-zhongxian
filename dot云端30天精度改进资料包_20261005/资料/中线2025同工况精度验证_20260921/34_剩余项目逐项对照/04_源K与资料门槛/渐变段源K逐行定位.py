"""源渐变段605行的空间、拓扑与当前计算连接映射；不推进水力时间步。"""
from pathlib import Path
import collections,csv,hashlib,json,os,re,subprocess,sys
import numpy as np
import openpyxl

O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
BOOK=R/'01_原始资料/中线全线建模数据.xlsx'
INFO=Path('/Users/gaobingxiang/code/hydroclaw0418/datasources/中线/中线组件信息.xlsx')
model=json.loads((B/'共享输入/C07.json').read_text())['中线全线']['baseData'];sections=model['sections'];channels=model['channels']
w=openpyxl.load_workbook(BOOK,read_only=True,data_only=True)
trans=list(w['渐变段'].values)[1:];siphons=list(w['倒虹吸'].values)[1:]
assert len(trans)==605 and {r[0]for r in trans}==set(range(1,606))
env={**os.environ,'NUMBA_CACHE_DIR':'/private/tmp/gate_fix_cache','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
py=R.parents[1]/'water/pipedream-runner/.venv/bin/python'
code='''from pathlib import Path
import sys,json
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'30_分通道方法逐项测试/03_稳态与分通道试验'))
from 试验水力工具 import build
e,s,c,f,p=build()
print(json.dumps(dict(discretization=c['channel_discretization'],links=c['links'].to_dict('records'),superlinks=c['superlinks'].to_dict('records'),siphons=c['siphons_info'],C_uk=e.model._C_uk.tolist(),C_dk=e.model._C_dk.tolist(),dx_uk=e.model._dx_uk.tolist(),dx_dk=e.model._dx_dk.tolist(),runtime_t=e.model.t),ensure_ascii=False))
'''
runtime=json.loads(subprocess.check_output([str(py),'-c',code,str(R)],env=env,text=True))
assert runtime['runtime_t']==0
meta=runtime['discretization'];links={r['id']:r for r in runtime['links']}

def name_only(s):return re.sub(r'-pool\d+$','',re.sub(r'^DHX\d*-','',s))
sr_to_runtime={}
for r in siphons:
    hits=[v for v in runtime['siphons'] if name_only(v['name'])==name_only(r[1]) and abs(v['inlet_zb']-r[3])<1e-6 and abs(v['outlet_zb']-r[7])<1e-6 and abs(v['physical_length']-r[13])<1e-4]
    if len(hits)!=1:
        hits=[v for v in runtime['siphons'] if name_only(v['name'])==name_only(r[1])]
    if len(hits)==1:sr_to_runtime[r[0]]=hits[0]
assert len(sr_to_runtime)==105,len(sr_to_runtime)

def point_x(p):
    a,b=p['source_interval'];fa=float(p['fraction'])
    return sections[a]['location']*(1-fa)+sections[b]['location']*fa

spatial={}
for name in channels:
    mm=meta[name];pp={p['junction_id']:p for p in mm['sections']}
    spatial[name]=[]
    for i in mm['internal_link_ids']:
        ll=links[i];pa,pb=pp[ll['j_0']],pp[ll['j_1']]
        xa,xb=point_x(pa),point_x(pb)
        spatial[name].append(dict(link_id=i,start_m=min(xa,xb),end_m=max(xa,xb),shape=ll['shape'],ctrl=ll['ctrl'],C=ll['C'],A_c=ll['A_c']))

def source_geometry(sn):
    yz=sections[sn]['yz'];z=min(p[1]for p in yz);bottoms=[p[0]for p in yz if abs(p[1]-z)<1e-8]
    b=max(bottoms)-min(bottoms);m1=(min(bottoms)-yz[0][0])/(yz[0][1]-z);m2=(yz[-1][0]-max(bottoms))/(yz[-1][1]-z)
    return [z,b,m1,m2]

mapped=[]
for row in trans:
    tid=int(row[0]);a,b=float(row[2])*1000,float(row[6])*1000
    exact=[];contains=[];starts=[];ends=[]
    for ch,v in channels.items():
        sn=v['sec_names'];lo=min(sections[s]['location']for s in sn);hi=max(sections[s]['location']for s in sn)
        sa=[s for s in sn if abs(sections[s]['location']-a)<1e-5];sb=[s for s in sn if abs(sections[s]['location']-b)<1e-5]
        starts.extend((ch,s)for s in sa);ends.extend((ch,s)for s in sb)
        if lo-1e-5<=a and b<=hi+1e-5:contains.append(ch)
        if len(sa)==1 and len(sb)==1:exact.append((ch,sa[0],sb[0]))
    selected=None
    if len(exact)==1:selected=exact[0];status='两端原断面唯一定位'
    elif len(exact)>1:status='两端重复需核对'
    elif len(contains)==1:selected=(contains[0],None,None);status='区间包含但原断面端点缺失'
    elif len(starts)==1 and not ends:status='仅渐变段起点存在'
    elif starts and ends:status='两端落在不同渠道或重复拓扑'
    else:status='区间未闭合'
    ch=selected[0] if selected else None
    overlaps=[]
    for cn in ([ch] if ch else sorted({z[0]for z in starts+ends+[(k,None)for k in contains]})):
        for q in spatial[cn]:
            ol=min(b,q['end_m'])-max(a,q['start_m'])
            if ol>1e-6:overlaps.append({**q,'channel':cn,'overlap_m':ol})
    endpoints=[]
    if ch:
        secnames=channels[ch]['sec_names'];lo=min(sections[s]['location']for s in secnames);hi=max(sections[s]['location']for s in secnames)
        if abs(a-lo)<1e-5:endpoints.append('源渠起点')
        if abs(b-hi)<1e-5:endpoints.append('源渠终点')
    adj=[]
    for sr in siphons:
        relation=[]
        if abs(b-sr[2]*1000)<1e-5:relation.append('渐变段接倒虹吸进口')
        if abs(a-sr[6]*1000)<1e-5:relation.append('倒虹吸出口接渐变段')
        if relation:
            rr=sr_to_runtime[sr[0]]
            adj.append(dict(source_id=sr[0],source_name=sr[1],source_cells=f'倒虹吸!A{sr[0]+1}:P{sr[0]+1}',relation=relation,
                current_name=rr['name'],superlink_id=rr['superlink_id'],explicit_K=rr['loss_coeff'],source_K=sr[15],physical_length_m=rr['physical_length'],
                source_barrel_length_m=sr[13],source_chainage_length_m=(sr[6]-sr[2])*1000,
                runtime_superlink=next(v for v in runtime['superlinks']if v['id']==rr['superlink_id']),
                runtime_dx_uk=runtime['dx_uk'][rr['superlink_id']],runtime_dx_dk=runtime['dx_dk'][rr['superlink_id']] ))
    geo=[]
    if selected and selected[1]:
        for edge,sn,vals in [('进口',selected[1],row[3:6]),('出口',selected[2],row[7:10])]:
            z,bb,m1,m2=source_geometry(sn);dv=[z-float(vals[0]),bb-float(vals[1]),m1-float(vals[2]),m2-float(vals[2])]
            geo.append(dict(edge=edge,section=sn,current=[z,bb,m1,m2],source=list(vals),diff=dv,match=max(map(abs,dv))<1e-5))
    mapped.append(dict(source_id=tid,source_name=row[1],source_cells=f'渐变段!A{tid+1}:L{tid+1}',start_m=a,end_m=b,length_m=b-a,
        inlet_z=row[3],inlet_b=row[4],inlet_m=row[5],outlet_z=row[7],outlet_b=row[8],outlet_m=row[9],K_column=row[10],L_column=row[11],
        K_L_semantics='未知；两列表头均为局部水力损失',velocity_head_reference='未知',mapping_status=status,current_channel=ch,
        exact_candidates=exact,containing_channels=contains,start_sections=starts,end_sections=ends,overlap_links=overlaps,
        covered_m=sum(v['overlap_m']for v in overlaps) if ch else None,candidate_overlap_m=sum(v['overlap_m']for v in overlaps),geometry_comparison=geo,source_channel_ends=endpoints,
        adjacent_explicit_siphons=adj,
        overlap_risk=('相邻倒虹吸已有显式K且两端另有默认项，系数物理覆盖边界未明，不能判定重复' if adj else '未直接邻接源倒虹吸；仍有所在SuperLink端部默认项，不能视为源K已导入'),
        source_coeff_mapped_to_control=any(q['ctrl']for q in overlaps),real_facility_coefficient_test_ready=False))

wi=openpyxl.load_workbook(INFO,read_only=True,data_only=True);ss=wi.active
component_trans={r[0]:(i,r) for i,r in enumerate(ss.values,1) if len(r)>1 and isinstance(r[1],str) and r[1].startswith('JBD')}
compdiff=[]
for r in mapped:
    i,t=component_trans[r['source_id']]
    if any((a!=b if a is None or b is None else abs(float(a)-float(b))>1e-8) for a,b in [(r['start_m']/1000,t[2]),(r['end_m']/1000,t[3]),(r['K_column'],t[4]),(r['L_column'],t[5])]):
        compdiff.append(dict(id=r['source_id'],component_row=i,component=list(t[:6]),source=[r[k]for k in ['source_name','start_m','end_m','K_column','L_column']]))
summary=dict(total=605,status_counts=dict(collections.Counter(r['mapping_status']for r in mapped)),
    K_nonzero=sum(bool(r['K_column'])for r in mapped),L_nonzero=sum(bool(r['L_column'])for r in mapped),
    K_missing=sum(r['K_column']is None for r in mapped),L_missing=sum(r['L_column']is None for r in mapped),
    adjacent_siphon_rows=sum(bool(r['adjacent_explicit_siphons'])for r in mapped),
    source_channel_end_rows=sum(bool(r['source_channel_ends'])for r in mapped),
    source_K_control_rows=sum(r['source_coeff_mapped_to_control']for r in mapped),
    exact_rows_source_geometry_all_match=sum(bool(r['geometry_comparison']) and all(g['match']for g in r['geometry_comparison'])for r in mapped),
    exact_rows_source_geometry_mismatch=sum(bool(r['geometry_comparison']) and not all(g['match']for g in r['geometry_comparison'])for r in mapped),
    component_table_diff_count=len(compdiff),physical_coefficient_ready_count=0,runtime_time_s=runtime['runtime_t'])
out=dict(summary=summary,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [BOOK,INFO,B/'共享输入/C07.json']},
    note='源K速度基准/两列物理定义仍未知。区间定位不是局损物理定义核验。未运行step、未改模型。',component_differences=compdiff,rows=mapped)
(O/'605渐变段源K映射.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
flat=[]
for r in mapped:
    flat.append({k:r[k]for k in ['source_id','source_name','source_cells','start_m','end_m','length_m','inlet_z','inlet_b','inlet_m','outlet_z','outlet_b','outlet_m','K_column','L_column','mapping_status','current_channel','covered_m','overlap_risk']}|
        dict(current_links=';'.join(str(v['link_id'])for v in r['overlap_links']),source_channel_ends=';'.join(r['source_channel_ends']),
             adjacent_siphons=';'.join(v['current_name']+f" K={v['explicit_K']}"for v in r['adjacent_explicit_siphons']),
             geometry_match=(all(v['match']for v in r['geometry_comparison']) if r['geometry_comparison']else '未闭合'),
             velocity_reference='未知',K_L_semantics='未知',physical_test_ready=False))
with (O/'605渐变段源K映射.csv').open('w',newline='',encoding='utf-8-sig')as f:
    ww=csv.DictWriter(f,fieldnames=list(flat[0]));ww.writeheader();ww.writerows(flat)
(O/'运行定位元数据.json').write_text(json.dumps({'siphons':runtime['siphons'],'open_channels':{k:{'superlink_id':meta[k]['superlink_id'],'C_uk':runtime['C_uk'][meta[k]['superlink_id']],'C_dk':runtime['C_dk'][meta[k]['superlink_id']], 'superlink':next(v for v in runtime['superlinks']if v['id']==meta[k]['superlink_id']), 'dx_uk':runtime['dx_uk'][meta[k]['superlink_id']], 'dx_dk':runtime['dx_dk'][meta[k]['superlink_id']]}for k in channels}},ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps([r for r in flat if r['mapping_status']!='两端原断面唯一定位'],ensure_ascii=False,indent=2))
w.close();wi.close()
