"""q20s1仅改源桩号的零步候选；不混入闸底、端面或西赵河修复。"""
from pathlib import Path
import ast
import copy
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
DEPS=B/'共享依赖'
P=ROOT/'41_基础模型深入检查与修复/03_结构转换核查/结构转换逐项核查.py'
sp=importlib.util.spec_from_file_location('meihe42_build',P);helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
import numpy as np
import pandas as pd

SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARRSHA=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
CHANNEL='POOL20-1-JJZ20DHX29'
STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')


def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def location_candidate(source):
    model=copy.deepcopy(source)
    assert model['baseData']['sections']['q20s1']['location']==385036
    assert model['baseData']['channels'][CHANNEL]['sec_names'][:2]==['q20s1','q20s2']
    assert model['baseData']['sections']['q20s2']['location']==385349
    model['baseData']['sections']['q20s1']['location']=385306.
    return model


scope=dict(np=np,pd=pd,re=re,Path=Path,R=ROOT)
names={'meshes','weights_at','Forcing','flow_operator','flow_operators'}
nodes=[n for n in ast.parse((DEPS/'水力运行脚本/试验公共.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert {n.name for n in nodes}==names
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DEPS/'水力运行脚本/试验公共.py'),'exec'),scope)


def water_ops(source,c,m):
    old=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text())
    result=[];mesh=scope['meshes'](source,c)
    for op in old:
        matches=[(n,me) for n,me in mesh.items() if set(op['indices']).issubset(set(me['ids']))]
        assert len(matches)==1
        name,me=matches[0]
        try:
            ids,w=scope['weights_at'](me,op['chainage'])
            result.append({**op,'channel':name,'indices':ids.tolist(),'weights':w.tolist(),'offset':float(m._z_inv_Ik[ids]@w),'valid':True})
        except ValueError as exc:result.append({**op,'channel':name,'valid':False,'reason':str(exc)})
    return result


def physical_routes(forcing,source,c):
    mesh=scope['meshes'](source,c)
    physical={int(i):(name,float(x)) for name,me in mesh.items() for x,i in zip(me['x'],me['ids'])}
    out=[]
    for r in forcing.routing:
        name=r['target'];values=forcing.bleeder_values.get(name,np.full(len(forcing.times),float(c['initial_data']['bleeders'].get(name,0))))
        out.append(dict(target=name,input_channel=source['baseData']['bleeders'][name]['channel'],mode=r['mode'],
            physical_support=[dict(channel=physical[i][0],chainage_m=physical[i][1],weight=w) for i,w in zip(r['junction_ids'],r['weights'])],
            actual_value_sha256=ARRSHA(values),nonzero_count=int(np.count_nonzero(values)),indices=r['junction_ids'],weights=r['weights']))
    return out


def main():
    source_path=B/'共享输入/C07.json'
    source=json.loads(source_path.read_text())['中线全线'];modified=location_candidate(source)
    patch=dict(base_source=str(source_path),base_sha256=SHA(source_path),patch_root='中线全线',
        patches=[dict(op='replace',path='/baseData/sections/q20s1/location',old=385036.,value=385306.)],
        evidence_cells=['建模表节制闸!B20:D20','建模表渠道!B205:D205','2023结构表节制闸!B22:D22'],
        source_target='梅河JZZ20闸385+306；CH204起点385.306km/终点385.349km；q20s2=385349m',
        scope='只恢复q20s1桩号，首段313m变43m；断面yz、渠道连接、节点底、门孔底、n/C和输入时序不变',
        observation_and_offtake_chainages_unchanged=True,hydraulic_run_authorized=False)
    save('梅河首断面桩号单字段候选.json',patch)
    e0,c0=helper.physical_build(source);e1,c1=helper.physical_build(modified);m0,m1=e0.model,e1.model
    with np.load(B/'共享输入/初始化后状态_0s.npz') as z:
        for m in [m0,m1]:
            for key in STATE_KEYS:getattr(m,key)[:]=z[key]
    assert not hasattr(m1,'_endpoint_geometry_binding')
    f0=scope['Forcing'](e0,source,c0);f1=scope['Forcing'](e1,modified,c1)
    w0,w1=water_ops(source,c0,m0),water_ops(modified,c1,m1)
    q0,a0=scope['flow_operators'](e0,source,c0);q1,a1=scope['flow_operators'](e1,modified,c1)
    r0,r1=physical_routes(f0,source,c0),physical_routes(f1,modified,c1)
    changed_w=[dict(before=a,after=b) for a,b in zip(w0,w1) if a!=b]
    qa,qb={x['station']:x for x in q0},{x['station']:x for x in q1}
    changed_q=[dict(station=n,before=qa.get(n),after=qb.get(n)) for n in qa.keys()|qb.keys() if qa.get(n)!=qb.get(n)]
    changed_r=[dict(before=a,after=b) for a,b in zip(r0,r1) if a!=b]
    meta0,meta1=c0['channel_discretization'][CHANNEL],c1['channel_discretization'][CHANNEL];k=meta0['superlink_id']
    assert k==meta1['superlink_id'];ids=np.flatnonzero(m0._ki==k)
    before_length=m0._dx_ik[ids];after_length=m1._dx_ik[ids]
    dx_changed=np.flatnonzero(m0._dx_ik!=m1._dx_ik)
    params=['_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk']
    geo=[f'_g{i}_ik' for i in range(1,8)]+['_z_inv_Ik','_z_inv_j','_J_uk','_J_dk','_Ik','_Ip1k','_geom_codes']
    assert all(np.array_equal(getattr(m0,k0),getattr(m1,k0)) for k0 in params+geo+list(STATE_KEYS))
    assert m0.M==m1.M and m0.NIk==m1.NIk and m0.Nik==m1.Nik
    v=[]
    for m in [m0,m1]:
        t=copy.deepcopy(m);t.link_hydraulic_geometry();t.compute_storage_volumes()
        v.append(float(np.sum(t._A_ik*t._dx_ik)+np.sum(t.V_j[~t.bc])))
    first_a=float(m0._A_ik[ids[0]])
    # 体积核验在各副本用同一六态重算，不使用构模缓存。
    t=copy.deepcopy(m0);t.link_hydraulic_geometry();first_a=float(t._A_ik[ids[0]])
    assert abs((v[1]-v[0])-first_a*(-270))<1e-5
    mesh0,mesh1=scope['meshes'](source,c0),scope['meshes'](modified,c1)
    keys0={(n,float(x)):int(i) for n,me in mesh0.items() for x,i in zip(me['x'],me['ids'])}
    keys1={(n,float(x)):int(i) for n,me in mesh1.items() for x,i in zip(me['x'],me['ids'])}
    endpoint_id=int(meta0['sections'][0]['junction_id'])
    # POOL20未反向，首断面的水力接口身份/底高程/截面不变；仅标错的桩号恢复。
    assert meta0['flow_sign']==meta1['flow_sign']==1 and meta1['sections'][0]['junction_id']==endpoint_id
    node=e0.node_index[source['baseData']['channels'][CHANNEL]['node1']]
    report=dict(status='ZERO_STEP_CHECKED',hydraulic_steps=0,source_hashes={str(p):SHA(p) for p in [Path(__file__),P,source_path,B/'共享输入/初始化后状态_0s.npz',DEPS/'水力运行脚本/试验公共.py',HERE/'梅河首断面桩号单字段候选.json']},
        only_input_field_changed='/baseData/sections/q20s1/location:385036→385306',
        array_counts=dict(superjunctions=m0.M,internal_junctions=m0.NIk,internal_links=m0.Nik,all_counts_unchanged=True),
        geometry=dict(channel=CHANNEL,superlink_id=k,source_first=385036,candidate_first=385306,second_section=385349,
            first_link_id=int(ids[0]),first_length_before_m=float(before_length[0]),first_length_after_m=float(after_length[0]),
            total_channel_length_before_m=float(before_length.sum()),total_channel_length_after_m=float(after_length.sum()),
            changed_internal_length_ids=dx_changed.tolist(),all_geometry_parameters_bed_and_connections_unchanged=True),
        physical_node_identity=dict(common_open_channel_node_keys=len(keys0.keys()&keys1.keys()),removed=list(keys0.keys()-keys1.keys()),added=list(keys1.keys()-keys0.keys()),
            endpoint_id_unchanged=endpoint_id,endpoint_attached_node=source['baseData']['channels'][CHANNEL]['node1']),
        water_operators=dict(before_count=len(w0),after_count=len(w1),invalid_after=[o for o in w1 if not o['valid']],changes=changed_w),
        flow_operators=dict(before_count=len(q0),after_count=len(q1),changes=changed_q,audit_after=a1),
        offtake_routes=dict(before_count=len(r0),after_count=len(r1),changes=changed_r,all_actual_source_series_unchanged=all(a['actual_value_sha256']==b['actual_value_sha256'] for a,b in zip(r0,r1))),
        forcing_exact_equality={key:bool(np.array_equal(getattr(f0,key),getattr(f1,key),equal_nan=True)) for key in ['times','h','q','q0','u']},
        initial_state_projection=dict(six_states_can_be_kept_exact=True,
            rationale='唯一移动点q20s1仍是同一NODE-main-38下游接口，床高/断面/水力连接均不变；错误仅为桩号与首段长度。原六态可按同水力接口身份保留，无需新增水位插值或状态校正。其余明渠物理位置完全相同。',
            caution='必须保留候选构模的新x_Ik/长度，不能从旧NPZ灌回x/A/V；算子按原测点/分水的绝对桩号重新映射，不跟随错误坐标平移。',
            endpoint_absolute_head_m=float(m0.h_Ik[endpoint_id]+m0._z_inv_Ik[endpoint_id]),
            adjacent_superjunction_head_m=float(m0.H_j[node]),
            original_initial_volume_m3=v[0],candidate_initial_volume_m3=v[1],initial_geometry_volume_difference_m3=v[1]-v[0],
            exact_volume_difference_formula='-270m × 同一初态首连接过水面积',first_link_initial_area_m2=first_a),
        isolated_from_other_repairs=dict(gate_floor_hook=False,endpoint_hook=False,western_connection_patch=False),
        next_step='只读映射结果交主进程；任何过程配对另行授权。')
    save('梅河首断面构模与算子差分.json',report)
    print(json.dumps({k:report[k] for k in ['status','array_counts','geometry','physical_node_identity','forcing_exact_equality']},ensure_ascii=False))
    print('operator changes',len(changed_w),len(changed_q),len(changed_r),'invalid water',len(report['water_operators']['invalid_after']))


if __name__=='__main__':main()
