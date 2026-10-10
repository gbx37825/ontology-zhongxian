"""42结构只读核查：源表→C07→实际数组，及三个已保存状态的域外端面分类。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']='/private/tmp/pipedream_frozen_41_numba'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
B=ROOT/'26_冻结基线与后续归因'
R41=ROOT/'41_基础模型深入检查与修复'
spec=importlib.util.spec_from_file_location('structure41_for42',R41/'03_结构转换核查/结构转换逐项核查.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
import numpy as np
import pandas as pd

SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def contour(sec):
    yz=np.asarray(sec['yz'],float);z=yz[:,1].min();h=yz[:,1].max()-z
    bottom=yz[np.abs(yz[:,1]-z)<1e-8,0];b=np.ptp(bottom)
    return dict(bottom_m=float(z),width_m=float(b),slope=float((np.ptp(yz[:,0])-b)/(2*h)),top_m=float(z+h),height_m=float(h))


def source_endpoints(index):
    records=[]
    for sheet in ['渠道','渐变段','倒虹吸']:
        for r in index['model_workbook']['sheets'][sheet]['rows']:
            v=r['cells'];row=r['row']
            for side in ['in','out']:
                x,z,b,m=(2 if side=='in' else 3,6 if side=='in' else 7,4,5) if sheet=='渠道' else ((2,3,4,5) if side=='in' else (6,7,8,9))
                if any(v[j] is None for j in [x,z,b,m]):continue
                item=dict(sheet=sheet,row=row,name=v[1],side=side,chainage_m=float(v[x])*1000,bottom_m=float(v[z]),width_m=float(v[b]),slope=float(v[m]))
                if sheet=='渠道':item['top_m']=float(v[8 if side=='in' else 9]);item['shape_code']=v[11]
                if sheet=='倒虹吸':item.update(barrels=v[10],single_width_m=v[11],single_height_m=v[12],barrel_length_m=v[13])
                records.append(item)
    return records


def matching_sources(section,ends):
    geom=contour(section);x=float(section['location']);matched=[]
    for e in ends:
        if abs(e['chainage_m']-x)>1e-6:continue
        matches=all(abs(e[k]-geom[k])<1e-6 for k in ['bottom_m','width_m','slope'])
        matched.append({**e,'bottom_width_slope_match':matches,
            'top_match':None if 'top_m' not in e else abs(e['top_m']-geom['top_m'])<1e-6})
    return matched


def adjacent_objects(bd,node):
    rows=[]
    for group in ['gates','siphons']:
        for name,v in bd[group].items():
            if node in [v.get('node1'),v.get('node2')]:
                rows.append(dict(type=group,name=name,position='node1' if node==v['node1'] else 'node2'))
    return rows


def classify(adjacency):
    kinds={r['type'] for r in adjacency}
    if 'gates' in kinds:return '节制闸相邻明渠端'
    if 'siphons' in kinds:return '已独立建模闭口结构的明渠渐变端'
    return '渠道外边界或其他节点'


def main():
    index=json.loads((HERE/'原表结构与版本索引.json').read_text())
    raw=json.loads((B/'共享输入/C07.json').read_text())['中线全线'];bd=raw['baseData'];ends=source_endpoints(index)
    engine,c=helper.physical_build(raw);m=engine.model
    node_names=list(c['superjunctions']['name'])
    assert len(node_names)==len(bd['nodes'])
    paths={'共同0s':B/'共享输入/初始化后状态_0s.npz',
        '参考S03第30天':R41/'10_端面几何锁参验证/reference/末态.npz',
        '端面修正第30天':R41/'10_端面几何锁参验证/endpoint/末态.npz'}
    watched=[Path(__file__),HERE/'原表结构与版本索引.json',B/'共享输入/C07.json',Path(helper.__file__),*paths.values()]
    hashes={str(p):SHA(p) for p in watched}
    selected=[];source_points=[]
    for name,ch in bd['channels'].items():
        match=re.match(r'^POOL(\d+)',name,re.I)
        if not match or not 46<=int(match[1])<=55:continue
        meta=c['channel_discretization'][name];k=meta['superlink_id'];sl=c['superlinks'].loc[k]
        source_positions=[float(bd['sections'][s]['location']) for s in ch['sec_names']]
        rows=[]
        for p in meta['sections']:
            if p['name'] is not None:
                sec=bd['sections'][p['name']]
                entry=dict(channel=name,section=p['name'],location_m=sec['location'],geometry=contour(sec),source_matches=matching_sources(sec,ends))
                source_points.append(entry)
            j=p['junction_id'];assert abs(m._z_inv_Ik[j]-p['z_inv'])<1e-9
        for row in c['links'].loc[c['links'].k==k].itertuples():
            assert abs(m._dx_ik[row.id]-row.dx)<1e-9
            assert all(abs(getattr(m,f'_g{i}_ik')[row.id]-getattr(row,f'g{i}'))<1e-9 for i in range(1,4))
        selected.append(dict(channel=name,pool=int(match[1]),source_node1=ch['node1'],source_node2=ch['node2'],
            source_first_section=ch['sec_names'][0],source_last_section=ch['sec_names'][-1],
            source_chainage_range_m=[source_positions[0],source_positions[-1]],source_length_m=float(np.abs(np.diff(source_positions)).sum()),
            actual_length_m=float(m._dx_ik[m._ki==k].sum()),solver_flow_sign=meta['flow_sign'],
            solver_start_node=node_names[int(m._J_uk[k])],solver_end_node=node_names[int(m._J_dk[k])],
            input_source_node1_bed_m=bd['nodes'][ch['node1']]['zb'],input_source_node2_bed_m=bd['nodes'][ch['node2']]['zb'],
            actual_solver_start_node_bed_m=float(m._z_inv_j[m._J_uk[k]]),actual_solver_end_node_bed_m=float(m._z_inv_j[m._J_dk[k]]),
            actual_solver_start_endpoint_bed_m=float(m._z_inv_uk[k]),actual_solver_end_endpoint_bed_m=float(m._z_inv_dk[k]),
            actual_start_offset_m=float(sl.in_offset),actual_end_offset_m=float(sl.out_offset),
            first_endpoint=meta['sections'][0],last_endpoint=meta['sections'][-1],
            link_shapes=c['links'].loc[c['links'].k==k,'shape'].value_counts().to_dict()))
    save('JZZ47至55实际连接与源断面.json',dict(source_hashes=hashes,hydraulic_steps=0,channels=selected,source_points=source_points,
         actual_tables_equal_model_arrays=True,known_exception_retained='POOL46-2槐河二源里程冲突为既知项，不计新增修复'))
    rows=[]
    for label,path in paths.items():
        with np.load(path) as z:
            for key in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']:getattr(m,key)[:]=z[key]
        for name,ch in bd['channels'].items():
            meta=c['channel_discretization'][name];k=meta['superlink_id']
            for side,point in [('u',meta['sections'][0]),('d',meta['sections'][-1])]:
                i=int(m._I_1k[k] if side=='u' else m._I_Np1k[k]);link=int(m._i_1k[k] if side=='u' else m._i_nk[k]);j=int(getattr(m,'_J_'+side+'k')[k]);zb=float(getattr(m,'_z_inv_'+side+'k')[k])
                assert i==point['junction_id'] and abs(zb-point['z_inv'])<1e-9
                h=float(m.h_Ik[i]);theta=float(m.H_j[j]>=zb);hj=float(theta*(m.H_j[j]-zb));oldtop=float(m._g1_ik[link]);newtop=float(point['height']);y=(h+hj)/2
                dry=min(h,hj)<-1e-9;old_exceeded=max(h,hj)>oldtop+1e-9 or dry;source_exceeded=max(h,hj)>newtop+1e-9 or dry
                reason='旧中点与真实端点都超出' if old_exceeded and source_exceeded else '仅旧中点高度判据超出' if old_exceeded else '仅真实端点高度判据超出' if source_exceeded else '两者均在范围内'
                node=node_names[j];adj=adjacent_objects(bd,node);sec=bd['sections'][point['name']];sources=matching_sources(sec,ends)
                matching=[v for v in sources if v['bottom_width_slope_match']]
                direct_tops=[v for v in matching if v.get('top_match') is True]
                a,b=int(m._Ik[link]),int(m._Ip1k[link]);link_depth=float((m.h_Ik[a]+m.h_Ik[b])/2)
                rows.append(dict(state=label,channel=name,side=side,superlink_id=k,source_section=point['name'],
                    source_chainage_m=float(sec['location']),adjacent_node=node,facility_category=classify(adj),adjacent_objects=adj,
                    source_shape=point['shape_kind'],source_bottom_m=zb,source_height_m=newtop,source_top_m=zb+newtop,
                    midpoint_height_m=oldtop,internal_endpoint_depth_m=h,effective_superjunction_depth_m=hj,mean_endpoint_depth_m=y,
                    current_source_endpoint_beyond=source_exceeded,old_midpoint_bound_beyond=old_exceeded,union_outside=old_exceeded or source_exceeded,
                    union_reason=reason,source_mean_clipped=y>newtop+1e-9,old_mean_clipped=y>oldtop+1e-9,
                    adjacent_real_internal_mean_depth_m=link_depth,adjacent_real_internal_mean_clipped=link_depth>oldtop+1e-9,
                    max_source_excess_m=max(h,hj)-newtop,original_source_top_status='原渠道表同桩号几何及顶高程完全对应' if direct_tops else '该端原表未给可对应的顶高程或存在几何冲突',
                    source_matches=sources))
    summaries={}
    for state in paths:
        g=[r for r in rows if r['state']==state];outside=[r for r in g if r['union_outside']]
        summaries[state]=dict(total_open_ends=len(g),union_outside=len(outside),source_end_itself_outside=sum(r['current_source_endpoint_beyond'] for r in g),
            only_old_midpoint_bound=sum(r['union_reason']=='仅旧中点高度判据超出' for r in g),
            only_old_midpoint_but_real_internal_link_inside=sum(r['union_reason']=='仅旧中点高度判据超出' and not r['adjacent_real_internal_mean_clipped'] for r in g),
            true_source_mean_clipped=sum(r['source_mean_clipped'] for r in g),
            by_facility=pd.Series([r['facility_category'] for r in outside]).value_counts().to_dict(),
            by_reason=pd.Series([r['union_reason'] for r in outside]).value_counts().to_dict(),
            by_source_top_evidence=pd.Series([r['original_source_top_status'] for r in outside]).value_counts().to_dict())
    ids=[{(r['channel'],r['side']) for r in rows if r['state']==state and r['union_outside']} for state in paths]
    common=set.intersection(*ids);union=set.union(*ids)
    save('域外端面来源分类.json',dict(scope='三个已保存快照逐端核查；全程逐步文件只有总数，不能据三个快照声称全部时刻端面身份不变',
        hydraulic_steps=0,source_hashes=hashes,summaries=summaries,three_snapshot_union_end_count=len(union),three_snapshot_common_end_count=len(common),
        interpretation='仅旧中点限高不能一律当端面修正后真实端面超限；源均深是否截高与两侧严格范围另列。独立闭口结构的相邻明渠端不能因同桩号而整体改闭口。',rows=rows))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['adjacent_objects','source_matches']} for r in rows]).to_csv(HERE/'三个快照端面范围与设施分类.csv',index=False)
    repairs=[];regional_gates=[]
    selected_nodes={r['source_node1'] for r in selected}|{r['source_node2'] for r in selected}
    for name,g in bd['gates'].items():
        match=re.match(r'^JZZ(\d+)-',name)
        if not match or not 47<=int(match[1])<=55:continue
        selected_nodes.update([g['node1'],g['node2']])
        i=engine.gate_index[name]
        actual=float(m._z_inv_j[m._J_uo[i]]+m._z_o[i])
        regional_gates.append(dict(name=name,source_gate_bottom_m=g['zb'],actual_gate_bottom_m=actual,
            difference_m=actual-g['zb'],upstream_node=g['node1'],downstream_node=g['node2']))
    for repair in c['topology_repairs']:
        if repair['object_name'] in selected_nodes:repairs.append(repair)
    actual_closed=[]
    for row in c['siphons_info']:
        k=row['superlink_id'];ids0=np.flatnonzero(m._ki==k)
        assert all(c['links'].loc[i,'shape']=='rect_closed' for i in ids0)
        actual_closed.append(dict(name=row['name'],shape=row['shape'],length_m=float(m._dx_ik[ids0].sum()),
            source_length_m=row['physical_length'],barrels=row['n_orifice'],total_width_m=float(m._g2_ik[ids0[0]]),
            expected_total_width_m=row['n_orifice']*row['width']))
    save('区域零床高与闭口类型核验.json',dict(hydraulic_steps=0,regional_automatic_bed_repairs=repairs,
        whole_network_unrepaired_zero_bed_nodes=[node_names[j] for j in range(m.M) if m._z_inv_j[j]==0],
        regional_gate_source_and_effective_bottom=regional_gates,
        closed_105=actual_closed,all_closed_shape_correct=True,
        note='源节点zb=0不等于运行仍用0；必须采用修复后的superjunction实际数组判断，不能重复报告已执行的零值修复。'))
    assert all(SHA(p)==v for p,v in hashes.items())
    print(json.dumps(dict(regional_channels=len(selected),regional_source_sections=len(source_points),summary=summaries,
        union=len(union),common=len(common),zero_bed_repairs=len(repairs),hydraulic_steps=0),ensure_ascii=False))


if __name__=='__main__':main()
