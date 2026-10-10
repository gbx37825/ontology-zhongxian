"""两组源表宽坡恢复的零步构模/算子核验；无水动力推进、无其他修复混入。"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']='/private/tmp/pipedream_frozen_41_numba'
HERE=Path(__file__).resolve().parent
P=HERE/'梅河首断面桩号与映射核验.py'
sp=importlib.util.spec_from_file_location('mapping42_for_width_slope',P)
shared=importlib.util.module_from_spec(sp);sp.loader.exec_module(shared)
import numpy as np

SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARRSHA=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
B=shared.B
STATE_KEYS=shared.STATE_KEYS
PARAMS=['_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk']
GEOMETRY_UNCHANGED=['_g1_ik','_g4_ik','_g5_ik','_g6_ik','_g7_ik','_dx_ik','_dx_uk','_dx_dk',
    '_z_inv_Ik','_z_inv_j','_z_inv_uk','_z_inv_dk','_J_uk','_J_dk','_Ik','_Ip1k','_ki','_geom_codes','x_Ik']


def apply(source,patch):
    assert patch['base_sha256']==SHA(B/'共享输入/C07.json')
    candidate=copy.deepcopy(source)
    for row in patch['patches']:
        parts=row['path'].strip('/').split('/');obj=candidate
        for part in parts[:-1]:obj=obj[int(part)] if isinstance(obj,list) else obj[part]
        key=int(parts[-1]) if isinstance(obj,list) else parts[-1]
        assert obj[key]==row['old']
        obj[key]=row['value']
    return candidate


def restore(m,values):
    m.load_state({key:values[key].copy() for key in STATE_KEYS},compute_hydraulic_geometries=True)
    m.compute_storage_volumes()
    m.t=0.;m.iter_count=0


def volume(m):
    trial=copy.deepcopy(m);trial.link_hydraulic_geometry();trial.compute_storage_volumes()
    return float(np.sum(trial._A_ik*trial._dx_ik)+np.sum(trial.V_j[~trial.bc]))


def main():
    source=json.loads((B/'共享输入/C07.json').read_text())['中线全线']
    with np.load(B/'共享输入/初始化后状态_0s.npz') as z:state={key:z[key].copy() for key in STATE_KEYS}
    e0,c0=shared.helper.physical_build(source);m0=e0.model;restore(m0,state)
    f0=shared.scope['Forcing'](e0,source,c0)
    w0=shared.water_ops(source,c0,m0);q0,_=shared.scope['flow_operators'](e0,source,c0)
    r0=shared.physical_routes(f0,source,c0);v0=volume(m0)
    plans=[('q22s14宽坡恢复候选.json','q22s14宽坡零步核验.json'),('q26四断面宽坡恢复候选.json','q26四断面宽坡零步核验.json')]
    reports=[]
    for name,outname in plans:
        p=HERE/name;patch=json.loads(p.read_text());modified=apply(source,patch)
        e1,c1=shared.helper.physical_build(modified);m1=e1.model;restore(m1,state)
        assert not hasattr(m1,'_endpoint_geometry_binding')
        assert (m0.M,m0.NIk,m0.Nik)==(m1.M,m1.NIk,m1.Nik)
        assert c0['channel_discretization'].keys()==c1['channel_discretization'].keys()
        for key in STATE_KEYS+tuple(PARAMS+GEOMETRY_UNCHANGED):assert np.array_equal(getattr(m0,key),getattr(m1,key)),key
        touched={r['section'] for r in patch['points']}
        for sec in source['baseData']['sections']:
            old,new=source['baseData']['sections'][sec],modified['baseData']['sections'][sec]
            if sec not in touched:assert old==new
            else:
                assert {k:v for k,v in old.items() if k!='yz'}=={k:v for k,v in new.items() if k!='yz'}
                assert [r[1] for r in old['yz']]==[r[1] for r in new['yz']]
        f1=shared.scope['Forcing'](e1,modified,c1)
        w1=shared.water_ops(modified,c1,m1);q1,_=shared.scope['flow_operators'](e1,modified,c1)
        r1=shared.physical_routes(f1,modified,c1)
        assert w0==w1 and q0==q1 and r0==r1
        assert len(w1)==43 and len(q1)==44 and len(r1)==128
        forcing={key:bool(np.array_equal(getattr(f0,key),getattr(f1,key),equal_nan=True)) for key in ['times','h','q','q0','u']}
        assert all(forcing.values())
        closed=m0._parallel_k
        for side in ['u','d']:
            for metric in ['A','Pe','R','B']:
                key='_'+metric+'_'+side+'k'
                assert np.array_equal(getattr(m0,key)[closed],getattr(m1,key)[closed]),key
        changed=np.flatnonzero((m0._g2_ik!=m1._g2_ik)|(m0._g3_ik!=m1._g3_ik))
        links=[]
        for i in changed:
            k=int(m0._ki[i]);channel=c0['superlinks'].loc[k,'name'];meta=c0['channel_discretization'][channel]
            p0=next(p for p in meta['sections'] if p['junction_id']==m0._Ik[i]);p1=next(p for p in meta['sections'] if p['junction_id']==m0._Ip1k[i])
            links.append(dict(id=int(i),channel=channel,length_m=float(m0._dx_ik[i]),
                endpoints=[p0['name'],p1['name']],source_intervals=[p0['source_interval'],p1['source_interval']],
                old_g2_m=float(m0._g2_ik[i]),new_g2_m=float(m1._g2_ik[i]),old_g3=float(m0._g3_ik[i]),new_g3=float(m1._g3_ik[i]),
                g1_m=float(m0._g1_ik[i]),old_initial_A_m2=float(m0._A_ik[i]),new_initial_A_m2=float(m1._A_ik[i])))
        v1=volume(m1)
        direct=float(np.sum((m1._A_ik[changed]-m0._A_ik[changed])*m0._dx_ik[changed]))
        assert abs((v1-v0)-direct)<1e-5
        endpoints=[]
        for row in patch['points']:
            sn=row['section'];cn=row['channel'];p1=next(p for p in c1['channel_discretization'][cn]['sections'] if p['name']==sn)
            assert abs(p1['bottom_width']-row['source_b_m'])<1e-9 and abs(p1['side_slope']-row['source_slope'])<1e-9
            endpoints.append(dict(section=sn,internal_node_id=p1['junction_id'],g2_m=p1['bottom_width'],g3=p1['side_slope'],
                 height_m=p1['height'],bed_m=p1['z_inv'],absolute_initial_head_m=float(m1.h_Ik[p1['junction_id']]+m1._z_inv_Ik[p1['junction_id']])))
        report=dict(status='PASSED_ZERO_STEP',group=patch['group'],hydraulic_steps=0,
            source_hashes={str(f):SHA(f) for f in [Path(__file__),P,p,B/'共享输入/C07.json',B/'共享输入/初始化后状态_0s.npz']},
            candidate_sha256=SHA(p),array_counts=dict(superjunctions=m1.M,internal_nodes=m1.NIk,internal_links=m1.Nik,indices_and_counts_unchanged=True),
            changed_g2_g3_link_count=len(changed),changed_g2_g3_links=links,source_section_geometry_after=endpoints,
            invariant_arrays_exact=list(STATE_KEYS)+PARAMS+GEOMETRY_UNCHANGED,
            source_fields_preserved=['全部yz[][1]','location','num_interp','x1/y1/x2/y2','拓扑','n/C'],
            water_43_operators_exact=True,flow_44_operators_exact=True,offtake_128_routes_and_values_exact=True,
            forcing_five_arrays_exact=forcing,closed_105_end_geometry_exact=True,
            initial_state_projection=dict(required=False,six_states_exact=True,
                reason='全部断面位置、节点/链接身份和床高保持不变；同一绝对H与h自然一致，不新增观测校正或插值。只重算新几何缓存，不灌旧A/V。',
                original_geometric_volume_m3=v0,candidate_geometric_volume_m3=v1,geometry_volume_change_m3=v1-v0,
                independent_link_area_sum_volume_change_m3=direct,volume_is_not_assimilation_water=True),
            no_other_candidates_combined=['q20桩号','梅河闸底','41端面hook','西赵河连接','S02储水宽度'],
            next_gate='主进程审核候选与初始化水量变化后才决定是否推进；本结果不说明精度或全域物理适用性通过。')
        (HERE/outname).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        reports.append(dict(group=patch['group'],result_file=str(HERE/outname),result_sha256=SHA(HERE/outname),
            candidate_sha256=SHA(p),changed_links=changed.tolist(),initial_volume_difference_m3=v1-v0))
    (HERE/'两组宽坡候选零步核验汇总.json').write_text(json.dumps(dict(status='PASSED',hydraulic_steps=0,groups=reports),ensure_ascii=False,indent=2))
    print(json.dumps(reports,ensure_ascii=False))


if __name__=='__main__':main()
