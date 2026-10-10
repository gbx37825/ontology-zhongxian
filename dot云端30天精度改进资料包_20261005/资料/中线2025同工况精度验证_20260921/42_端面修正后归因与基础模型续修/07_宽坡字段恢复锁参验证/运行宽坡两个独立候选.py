"""42/07两个固定宽坡候选各自无同化推进；参考只读复用41/10。

只改已审核断面yz横坐标；无q20桩号、端面、闸底、西赵河或S02修复。
必须经主进程审查本入口与协议后，才能启动各自一次30天。
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
R42=HERE.parent
R=R42.parent
R41=R/'41_基础模型深入检查与修复'
B=R/'26_冻结基线与后续归因'
DEPS=B/'共享依赖'
PAIR10=R41/'10_端面几何锁参验证'
REFERENCE=PAIR10/'reference'
REFERENCE06=R41/'06_闸流水量一致性修复/G00_更严闸流一致性'
RUNNER10=PAIR10/'运行端面几何配对.py'
HELPER01=R41/'01_求解器核查/面积储水与闸流最小复现.py'
DIR02=R42/'02_退化渠段结构核查'
CASES={
    'q22':dict(patch='q22s14宽坡恢复候选.json',audit='q22s14宽坡零步核验.json',
        patch_sha256='f2ae308fa181642c5750e87e430943240f6ffec6df52fcf5f0a2cf7ebf002e4d',
        sections=['q22s14'],links=[1067,1068]),
    'q26':dict(patch='q26四断面宽坡恢复候选.json',audit='q26四断面宽坡零步核验.json',
        patch_sha256='da556446bc2c180c69f5753b438afec164abb084d9ff935d01f14964ed87ccf5',
        sections=['q26s1','q26s2','q26s3','q26s4'],links=list(range(1187,1194))),
}
STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
PARAM_KEYS=('_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk')
INTERNAL_GEOMETRY_KEYS=('_g1_ik','_g2_ik','_g3_ik','_g4_ik','_g5_ik','_g6_ik','_g7_ik',
                        '_dx_ik','_dx_uk','_dx_dk','_z_inv_Ik','_z_inv_j')
EXTENDED_FIXED_GEOMETRY_KEYS=('_x_Ik','_Ik','_Ip1k','_ki','_J_uk','_J_dk','_J_uo','_J_do',
                              '_z_inv_uk','_z_inv_dk','_z_o','_geom_codes')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


common10=load(RUNNER10,'_42_width_slope_common10')
dump=common10.dump
runtime_identity=common10.runtime_identity
contour_metadata=common10.contour_metadata
contour_values=common10.contour_values
residuals=common10.residuals


def build_runtime(case):
    shared=load(HELPER01,'_42_width_slope_build_source')
    plan=CASES[case];patch_path=DIR02/plan['patch']
    assert sha(patch_path)==plan['patch_sha256']
    base_source=json.loads((B/'共享输入/C07.json').read_text())['中线全线']
    source=copy.deepcopy(base_source);patch=json.loads(patch_path.read_text())
    assert patch['base_sha256']==sha(B/'共享输入/C07.json') and patch['patch_root']=='中线全线'
    assert [p['section'] for p in patch['points']]==plan['sections']
    expected_patches=[]
    for point in patch['points']:
        name=point['section'];old=source['baseData']['sections'][name]['yz'];new=point['candidate_yz']
        assert old==point['old_yz'] and len(old)==len(new)==4
        assert [p[1] for p in old]==[p[1] for p in new] and old[0]==new[0]
        for i in range(1,4):
            expected_patches.append(dict(op='replace',path=f'/baseData/sections/{name}/yz/{i}/0',
                                         old=old[i][0],value=new[i][0]))
        source['baseData']['sections'][name]['yz']=copy.deepcopy(new)
    assert patch['patches']==expected_patches
    inverse=copy.deepcopy(source)
    for name in plan['sections']:inverse['baseData']['sections'][name]['yz']=copy.deepcopy(base_source['baseData']['sections'][name]['yz'])
    assert inverse==base_source
    common=shared.bind(DEPS/'水力运行脚本/试验公共.py',
         ['Forcing','meshes','weights_at','flow_operator','flow_operators','hydraulic_volume'],
         dict(np=np,pd=pd,re=shared.re,Path=Path,R=R),'_42_width_slope_frozen_common')
    gate_max={k:float(v.get('b',1) or 1) for k,v in source['baseData']['gates'].items()}
    gate_max.update(json.loads((B/'共享输入/2025开度归一化上限.json').read_text()))

    def candidate_full_build(*args,gate_max_normalization=None,**kwargs):
        # 与41/01原构造仅source来源不同；物理管长及并联修复继续由26原函数执行。
        assert not args and not kwargs and gate_max_normalization==gate_max
        c=shared.parse_wxq_json(source,gate_max_openings=gate_max,
                                section_geometry_mode='safe',minimum_initial_depth_m=7.)
        c['superjunctions'].loc[:,'h_0']=7.
        c['superjunctions'].loc[:,'max_depth']=np.maximum(c['superjunctions'].max_depth.to_numpy(),7.)
        co=json.loads((B/'共享输入/Pipedream闸流系数.json').read_text())
        c['orifices'].loc[:,'C']=c['orifices'].name.map(co).fillna(c['orifices'].C)
        engine=shared.PipedreamHydraulicEngine(c,dt=15.,backend='numba');engine.model._method='lsq'
        return engine,source,c

    physics=shared.bind(DEPS/'并联结构物理修复.py',
         ['_correct_section','_link_geometry','_up_geometry','_down_geometry','_link_coeffs','attach_parallel_physics','build_engine_physics'],
         dict(np=np,types=shared.types,copy=copy,hashlib=hashlib,Path=Path,R=R,
              PipedreamHydraulicEngine=shared.PipedreamHydraulicEngine,_legacy_build=candidate_full_build),
         '_42_width_slope_frozen_physics')
    advance=shared.bind(DEPS/'参数并行运行脚本/常驻水力进程.py',['advance'],
                        dict(np=np,STATE_KEYS=STATE_KEYS),'_42_width_slope_original_advance').advance
    restore=shared.bind(DEPS/'参数并行运行脚本/局地参数EnKF.py',['restore'],{},'_42_width_slope_restore').restore
    e,s,c=physics.build_engine_physics(gate_max_normalization=gate_max)
    # 只转换原表以复核测点/分水，不构造参考求解器，不重复推进参考过程。
    baseline_c=shared.parse_wxq_json(base_source,gate_max_openings=gate_max,
                                    section_geometry_mode='safe',minimum_initial_depth_m=7.)
    assert len(c['links'])==len(baseline_c['links'])==2520
    assert len(c['junctions'])==len(baseline_c['junctions'])==2751
    assert len(c['superjunctions'])==len(baseline_c['superjunctions'])==291
    for key,columns in [('junctions',['id','k','z_inv']),('superjunctions',['id','name','z_inv']),
                        ('links',['id','j_0','j_1','k','n','shape','g1','g4']),
                        ('superlinks',['id','name','sj_0','sj_1','in_offset','out_offset'])]:
        assert c[key][columns].equals(baseline_c[key][columns]),key
    # 与26原物理管长同算式恢复基准转换表，用于坐标/长度精确对照，不构造参考求解器。
    for row in baseline_c['siphons_info']:
        k=int(row['superlink_id']);ids=baseline_c['links'].k==k
        old=float(baseline_c['links'].loc[ids,'dx'].sum());factor=float(row['physical_length'])/old
        baseline_c['links'].loc[ids,'dx']*=factor
        baseline_c['superlinks'].loc[baseline_c['superlinks'].id==k,'dx']=row['physical_length']
        for point in baseline_c['channel_discretization']['siphon_'+row['name']]['sections']:point['distance']*=factor
    return shared,common,advance,restore,e,s,c,base_source,baseline_c


def verify_geometry_and_operators(case,audit,m,e,source,c,base_source,baseline_c,common,oldcfg,out):
    original=oldcfg['internal_geometry_sha256'];plan=CASES[case]
    for key in INTERNAL_GEOMETRY_KEYS:
        if key not in ['_g2_ik','_g3_ik']:assert arrsha(getattr(m,key))==original[key],key
    assert [r['id'] for r in audit['changed_g2_g3_links']]==plan['links']
    base_arrays={key:getattr(m,key).copy() for key in PARAM_KEYS+INTERNAL_GEOMETRY_KEYS}
    for key,old_field,new_field in [('_g2_ik','old_g2_m','new_g2_m'),('_g3_ik','old_g3','new_g3')]:
        for row in audit['changed_g2_g3_links']:
            i=row['id'];assert getattr(m,key)[i]==row[new_field]
            base_arrays[key][i]=row[old_field]
        assert np.flatnonzero(base_arrays[key]!=getattr(m,key)).tolist()==plan['links']
        assert arrsha(base_arrays[key])==original[key]
    assert not set(plan['links'])&set(m._parallel_i.tolist()) and len(m._parallel_k)==105
    assert not hasattr(m,'_endpoint_geometry_binding')
    assert source['baseData']['sections']['q20s1']['location']==385036
    meihe=c['orifices'].name.str.startswith('JZZ20-')
    assert (c['orifices'].loc[meihe,'z_o']==0).all()
    assert (m._z_o[c['orifices'].loc[meihe,'id'].to_numpy(int)]==0).all()
    assert c['links'].drop(columns=['g2','g3']).equals(baseline_c['links'].drop(columns=['g2','g3']))
    for key,col in [('_Ik','j_0'),('_Ip1k','j_1'),('_ki','k')]:
        assert np.array_equal(getattr(m,key),baseline_c['links'][col].to_numpy())
    for key,col in [('_J_uk','sj_0'),('_J_dk','sj_1')]:
        assert np.array_equal(getattr(m,key),baseline_c['superlinks'][col].to_numpy())
    reference_x=np.zeros_like(m._x_Ik)
    reference_x[~m._is_start]=baseline_c['links'].groupby('k')['dx'].cumsum().values+m._dx_uk[m._ki]
    reference_x[m._is_start]=m._dx_uk
    assert np.array_equal(m._x_Ik,reference_x)
    base_arrays.update({key:getattr(m,key).copy() for key in EXTENDED_FIXED_GEOMETRY_KEYS})
    np.savez_compressed(out/'参考参数与内部几何数组.npz',**base_arrays)
    np.savez_compressed(out/'候选构造几何与连接数组.npz',
         **{key:getattr(m,key).copy() for key in INTERNAL_GEOMETRY_KEYS+EXTENDED_FIXED_GEOMETRY_KEYS})
    # 只在副本恢复参考g2/g3并重算瞬时几何，核对闭口端几何和初态面积；没有参考水力步。
    reference_geometry=copy.deepcopy(m)
    reference_geometry._g2_ik=base_arrays['_g2_ik'].copy()
    reference_geometry._g3_ik=base_arrays['_g3_ik'].copy()
    reference_geometry.link_hydraulic_geometry();reference_geometry.upstream_hydraulic_geometry()
    reference_geometry.downstream_hydraulic_geometry();reference_geometry.compute_storage_volumes()
    assert abs(common.hydraulic_volume(copy.deepcopy(reference_geometry))-oldcfg['initial_geometric_volume_m3'])<1e-5
    for side in ['u','d']:
        for metric in ['A','Pe','R','B']:
            key='_'+metric+'_'+side+'k'
            assert np.array_equal(getattr(m,key)[m._parallel_k],getattr(reference_geometry,key)[m._parallel_k])
    areas=[]
    for row in audit['changed_g2_g3_links']:
        i=row['id'];a0=float(reference_geometry._A_ik[i]);a1=float(m._A_ik[i]);dx=float(m._dx_ik[i])
        assert abs(a0-row['old_initial_A_m2'])<1e-9 and abs(a1-row['new_initial_A_m2'])<1e-9
        areas.append(dict(link_id=i,length_m=dx,original_area_m2=a0,candidate_area_m2=a1,volume_change_m3=(a1-a0)*dx))
    dump(out/'初始连接面积差.json',dict(rows=areas,independent_volume_change_m3=sum(r['volume_change_m3'] for r in areas),
         reference_geometry_only_reconstructed_on_copy=True,reference_hydraulic_steps=0))
    f=common.Forcing(e,source,c);f0=common.Forcing(e,base_source,baseline_c)
    hashes={key:arrsha(getattr(f,key)) for key in ['times','h','q','q0','u']}
    assert hashes==oldcfg['forcing_array_sha256']
    assert all(np.array_equal(getattr(f,k),getattr(f0,k),equal_nan=True) for k in hashes)
    assert len(f.routing)==len(f0.routing)==128 and f.routing==f0.routing
    assert f.bleeder_values.keys()==f0.bleeder_values.keys()
    assert all(np.array_equal(v,f0.bleeder_values[k]) for k,v in f.bleeder_values.items())
    frozen_ops=json.loads((REFERENCE/'实际位置观测算子.json').read_text());ops=[]
    mesh=common.meshes(source,c)
    for op in frozen_ops:
        matches=[v for v in mesh.values() if set(op['indices']).issubset(set(v['ids']))]
        assert len(matches)==1 and op['state']=='h_Ik'
        ids,weights=common.weights_at(matches[0],op['chainage'])
        actual={**op,'indices':ids.tolist(),'weights':weights.tolist(),'offset':float(m._z_inv_Ik[ids]@weights)}
        assert actual==op
        ops.append(actual)
    flow_ops,flow_audit=common.flow_operators(e,source,c)
    original_flow,original_audit=common.flow_operators(e,base_source,baseline_c)
    saved=json.loads((REFERENCE/'测流位置算子与适用范围.json').read_text())
    assert len(ops)==43 and len(flow_ops)==44
    assert flow_ops==original_flow==saved['operators'] and flow_audit==original_audit==saved['audit']
    manifest=dict(case=case,source_patch=str(DIR02/plan['patch']),source_patch_sha256=plan['patch_sha256'],
         source_sections=plan['sections'],only_source_fields_changed='指定断面yz[1:4][0]；全部z和其他源字段不变',
         only_g2_g3_changed_link_ids=plan['links'],all_other_geometry_and_connections_exact=True,
         x_Ik_exact_reference=True,reference_arrays_hash_verified=True,reference_x_cache_loaded=False,
         closed_105_end_geometry_exact=True,water_operators_43_exact=True,flow_operators_44_exact=True,
         offtake_routes_128_exact=True,forcing_five_arrays_exact=True,q20_chainage_candidate=False,
         endpoint_hook=False,gate_bottom_hook=False,western_connection_repair=False,S02_storage_candidate=False)
    dump(out/'源几何与算子身份核验.json',manifest)
    dump(out/'分水路由身份.json',dict(routes=f.routing,unchanged_against_original_source=True))
    return f,hashes,ops,flow_ops,flow_audit,manifest


def run_branch(case):
    assert case in CASES
    out=HERE/case;out.mkdir(parents=True,exist_ok=False);(out/'逐步残差分块').mkdir()
    started=time.time();shared=None;m=None
    status=dict(status='INITIALIZING',case=case,pid=os.getpid(),completed_cycles=0,completed_successful_substeps=0,
                rejected_attempts=0,model_time_s=0.,full_30day_completed=False,baseline_replaced=False)
    dump(out/'进度.json',status);print(json.dumps(status),flush=True)
    chunks=[];scalars=[];node_residual_buffer=[];node_rows=[];mapped_rows=[];meter_rows=[];cv_rows=[];network_rows=[];domain_rows=[]
    success=rejected=0;maxcv=maxnode=maxinternal=maxnetwork=0.;cv_bad=node_bad=0;iterations={}
    failure_handle=(out/'减步失败尝试.jsonl').open('w')

    def persist():
        nonlocal scalars,node_residual_buffer
        if scalars:
            frame=pd.DataFrame(scalars);p=out/'逐步残差分块'/f'分块_{len(chunks)+1:03d}.npz'
            data={key:frame[key].to_numpy() for key in frame};data['nonboundary_node_residual_m3s']=np.stack(node_residual_buffer)
            np.savez_compressed(p,**data)
            first=sum(v['rows'] for v in chunks)
            chunks.append(dict(file=str(p),sha256=sha(p),rows=len(frame),first_step=first,last_step_exclusive=first+len(frame),
                         start_s=float(frame.interval_start_s.iloc[0]),end_s=float(frame.time_s.iloc[-1])))
            scalars=[];node_residual_buffer=[]
            dump(out/'逐成功子步文件索引.json',dict(chunks=chunks,total_rows=sum(v['rows'] for v in chunks),
                 nonboundary_node_metadata='非定水位主节点索引.json',scalar_and_node_rows_aligned=True))
        for name,rows in [('半小时水位流量',node_rows),('实际位置水位',mapped_rows),('测流位置算子流量',meter_rows),
                         ('半小时测流至闸水量账',cv_rows),('全网半小时水量账',network_rows),('半小时原始与端面轮廓范围',domain_rows)]:
            if rows:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)

    try:
        protocol_path=HERE/'运行前约定.json';protocol=json.loads(protocol_path.read_text())
        assert protocol['flow_tol_m3s']==1e-9 and protocol['head_tol_m']==.0015 and protocol['num_iter']==20
        plan=CASES[case];audit_path=DIR02/plan['audit'];patch_path=DIR02/plan['patch']
        audit=json.loads(audit_path.read_text());assert audit['status']=='PASSED_ZERO_STEP'
        assert audit['candidate_sha256']==sha(patch_path)==plan['patch_sha256']
        assert [r['id'] for r in audit['changed_g2_g3_links']]==plan['links']
        assert all(audit['forcing_five_arrays_exact'].values())
        assert all(audit[k] for k in ['water_43_operators_exact','flow_44_operators_exact',
                                      'offtake_128_routes_and_values_exact','closed_105_end_geometry_exact'])
        oldcfg=json.loads((REFERENCE/'运行配置.json').read_text())
        old06cfg=json.loads((REFERENCE06/'运行配置.json').read_text())
        oldresult=json.loads((REFERENCE/'结果.json').read_text())
        oldexact=json.loads((REFERENCE/'与06逐值复现核验.json').read_text())
        assert oldresult['status']=='COMPLETED' and oldresult['full_30day_completed'] and oldexact['all_exact']
        assert not oldcfg['endpoint_hook_applied'] and oldcfg['parameter_sha256']==old06cfg['parameter_sha256']
        sources={**oldcfg['source_hashes'],**protocol['source_sha256'],str(protocol_path):sha(protocol_path),
                 str(REFERENCE/'运行配置.json'):sha(REFERENCE/'运行配置.json')}
        assert all(sha(p)==h for p,h in sources.items())
        shared,common,advance,restore,e,source,c,base_source,baseline_c=build_runtime(case);m=e.model
        own_x=m._x_Ik.copy();own_dx=m._dx_ik.copy()
        dump(out/'候选源输入.json',{'中线全线':source})
        with np.load(B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{key:z[key].copy() for key in STATE_KEYS})
        assert np.array_equal(m._x_Ik,own_x) and np.array_equal(m._dx_ik,own_dx)
        m.t=0.;m.iter_count=0
        initial={key:arrsha(getattr(m,key)) for key in STATE_KEYS}
        fixed={key:arrsha(getattr(m,key)) for key in PARAM_KEYS}
        internal_geometry={key:arrsha(getattr(m,key)) for key in INTERNAL_GEOMETRY_KEYS+EXTENDED_FIXED_GEOMETRY_KEYS}
        assert initial==oldcfg['initial_state_sha256'] and fixed==oldcfg['parameter_sha256']
        assert np.count_nonzero(m._dx_uk)==np.count_nonzero(m._dx_dk)==np.count_nonzero(m._A_SIk)==0 and m.n_w==m.n_p==0
        f,forcing_hashes,ops,flow_ops,flow_audit,manifest=verify_geometry_and_operators(
             case,audit,m,e,source,c,base_source,baseline_c,common,oldcfg,out)
        assert {key:arrsha(getattr(m,key)) for key in STATE_KEYS}==initial
        assert all(arrsha(getattr(m,key))==value for key,value in fixed.items())
        assert all(arrsha(getattr(m,key))==value for key,value in internal_geometry.items())
        dump(out/'实际位置观测算子.json',ops)
        dump(out/'测流位置算子与适用范围.json',dict(operators=flow_ops,audit=flow_audit,
             qualification='与41/10 reference同一冻结测点位置口径；端部最近连接中心近似保留，不宣称44点均为严格同坐标仪器真值。'))
        groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int))
                for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};assert len(groups)==59
        qop=next(o for o in flow_ops if o['station']=='JZZ16');assert qop==old06cfg['flow_operator']
        qids,qw=np.asarray(qop['indices']),np.asarray(qop['weights'])
        g00=json.loads((R/'38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数/运行配置.json').read_text())
        gids=np.asarray(g00['C_indices'])
        W=np.load(R/'30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy')
        jids,iids=np.flatnonzero(W[:m.M]),np.flatnonzero(W[m.M:]);wj,wi=W[jids],W[m.M+iids]
        assert arrsha(W)==old06cfg['control_volume_weights_sha256']
        node_ids=np.flatnonzero(~m.bc)
        dump(out/'非定水位主节点索引.json',dict(ids=node_ids.tolist(),names=[e.node_names[i] for i in node_ids],count=len(node_ids),units='m3/s'))
        heights,open_links,endpoints,endpoint_arrays=contour_metadata(m,source,c)
        dump(out/'原始与端面轮廓定义.json',dict(endpoints=endpoints,open_internal_node_ids=np.flatnonzero(np.isfinite(heights)).tolist(),
             open_internal_link_ids=open_links.tolist(),definition='接受态固定域诊断；端部有效平均水深仍为(h_internal+theta*(H_j-z_bed))/2',
             strict_endpoint_domain='max(h_internal,theta*h_super)<=min(g1_adjacent,g1_source)+1e-9 and min(effective_depths)>=-1e-9',
             raw_depth_diagnostic='另记录未乘theta的h_super=H_j-z_bed；raw域外/负水深计数不替代09有效水深口径',
             dynamic_geometry_switch=False,range_reporting_tolerance_m=1e-9,range_is_not_engineering_crest=True))
        volume=lambda model:common.hydraulic_volume(copy.deepcopy(model))
        initial_volume=volume(m);geometric_volume=initial_volume
        original_initial_volume=oldcfg['initial_geometric_volume_m3']
        initial_volume_delta=initial_volume-original_initial_volume
        expected_delta=audit['initial_state_projection']['geometry_volume_change_m3']
        assert abs(initial_volume_delta-expected_delta)<1e-5
        initial_areas=json.loads((out/'初始连接面积差.json').read_text())
        assert abs(initial_volume_delta-initial_areas['independent_volume_change_m3'])<1e-5
        dump(out/'初始几何容积差.json',dict(reference_initial_volume_m3=original_initial_volume,
             candidate_initial_volume_m3=initial_volume,candidate_minus_reference_m3=initial_volume_delta,
             changed_link_areas=initial_areas['rows'],formula='sum((A_candidate-A_reference)*dx)；同一六态下比较',
             status='SOURCE_WIDTH_SLOPE_CHANGE_AT_SAME_SIX_STATES',
             artificial_water_compensation=False,not_a_dynamic_water_balance_residual=True))
        identity=runtime_identity();assert not identity['numba_jit_disabled'] and identity['momentum_compiled_signatures']>0
        dump(out/'运行配置.json',dict(case=case,source_hashes=sources,protocol=protocol,
             initial_state_sha256=initial,parameter_sha256=fixed,forcing_array_sha256=forcing_hashes,
             internal_geometry_sha256=internal_geometry,initial_geometric_volume_m3=initial_volume,
             volume_scope='内部连接A*L及非定水位主节点体积；在副本刷新几何计量，不回灌',
             width_slope_manifest=manifest,reference_geometry_sha256=oldcfg['internal_geometry_sha256'],endpoint_geometry_manifest=None,endpoint_hook_applied=False,reference=str(REFERENCE),reference_06=str(REFERENCE06),
             assimilation=False,state_corrections=0,parameters_fixed=True,six_initial_states_match_06=True,
             source_geometry_changed=True,internal_midpoint_cross_section_changed=True,internal_length_changed=False,
             candidate_source_file_sha256=sha(out/'候选源输入.json'),initial_volume_delta_m3=initial_volume_delta,
             runtime_identity=identity,baseline_replaced=False))
        np.savez_compressed(out/'实际初始化_0s.npz',**{key:getattr(m,key) for key in STATE_KEYS})
        initial_domain=contour_values(m,heights,open_links,endpoint_arrays)
        reference_initial_domain=json.loads((REFERENCE/'初始轮廓范围.json').read_text())
        assert initial_domain==reference_initial_domain
        assert initial_domain['endpoint_outside_either_contour_count']==78
        dump(out/'初始轮廓范围.json',initial_domain)
        native_step=m.step;integrals={}

        def observed_step(**kw):
            nonlocal success,rejected,maxcv,maxnode,maxinternal,maxnetwork,cv_bad,node_bad
            t0,dt=float(m.t),float(kw['dt']);before={key:getattr(m,key).copy() for key in STATE_KEYS}
            kw.update(flow_tol=1e-9,head_tol=.0015);assert kw['num_iter']==20 and kw['strict'] is True
            try:native_step(**kw)
            except BaseException as exc:
                rejected+=1;exact=all(np.array_equal(getattr(m,key),before[key]) for key in STATE_KEYS)
                event=dict(attempt=rejected,start_s=t0,dt_s=dt,error=repr(exc),rolled_back_time_s=float(m.t),
                     six_state_rollback_exact=exact,iterations=int(getattr(m,'iter_elapsed',0)),
                     last_head_residual_m=float(getattr(m,'last_head_residual',0)),last_gate_residual_m3s=float(getattr(m,'last_gate_flow_residual',0)))
                for key in ['last_head_residual_m','last_gate_residual_m3s']:
                    if not np.isfinite(event[key]):event[key]=str(event[key])
                failure_handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');failure_handle.flush()
                dump(out/'最近失败尝试.json',event)
                if not exact or m.t!=t0:raise RuntimeError('原失败回退未恢复六态/时钟') from exc
                raise
            assert m.t==t0+dt and all(np.isfinite(getattr(m,key)).all() for key in STATE_KEYS)
            row,nr=residuals(m,before,kw,dt,W,jids,iids,wj,wi,qids,qw,gids,node_ids)
            maxcv=max(maxcv,abs(row['continuity_residual_m3s']));maxnode=max(maxnode,row['max_nonboundary_node_residual_m3s'])
            maxinternal=max(maxinternal,row['max_internal_residual_m3s']);maxnetwork=max(maxnetwork,abs(row['full_network_discrete_residual_m3s']))
            cv_bad+=int(abs(row['continuity_residual_m3s'])>1e-6);node_bad+=int(row['max_nonboundary_node_residual_m3s']>1e-6)
            used=int(m.iter_elapsed);iterations[str(used)]=iterations.get(str(used),0)+1;success+=1
            row.update(interval_start_s=t0,time_s=float(m.t),dt_s=dt,iterations=used,
                 head_iteration_residual_m=float(m.last_head_residual),gate_iteration_residual_m3s=float(m.last_gate_flow_residual),
                 **contour_values(m,heights,open_links,endpoint_arrays))
            scalars.append(row);node_residual_buffer.append(nr)
            integrals['duration']+=dt;integrals['steps']+=1
            for key,value in [('meter',row['Q16_cut_m3s']*dt),('gate',row['Q16_gate_m3s']*dt),
                    ('take',row['offtake_volume_m3']),('cv_storage',row['native_storage_change_m3']),
                    ('global_flux',row['net_boundary_input_m3']),('global_storage',row['full_network_discrete_storage_change_m3'])]:integrals[key]+=value
            integrals['global_max_residual']=max(integrals['global_max_residual'],abs(row['full_network_discrete_residual_m3s']))

        m.step=observed_step;retries=0;status.update(status='RUNNING',initialization_and_provenance_verified=True)
        dump(out/'进度.json',status)
        for cycle in range(1440):
            t0,t1=cycle*1800.,(cycle+1)*1800.
            integrals={key:0. for key in ['duration','steps','meter','gate','take','cv_storage','global_flux','global_storage','global_max_residual']}
            retry,_,_=advance(m,f,t0,t1);retries+=retry;assert integrals['duration']==1800. and m.t==t1
            for station,(ju,jd,ids) in groups.items():node_rows.append(dict(time_s=t1,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
            for op in ops:mapped_rows.append(dict(time_s=t1,station=op['station'],metric=op['metric'],mapped_sim=float(getattr(m,op['state'])[op['indices']]@np.asarray(op['weights'])+op.get('offset',0)),group='旧参与同化测点' if op['assimilated'] else '旧留出检验测点',chainage_m=op['chainage']))
            for op in flow_ops:meter_rows.append(dict(time_s=t1,station=op['station'],chainage_m=op['chainage'],model_flow_m3s=float(getattr(m,op['state'])[op['indices']]@np.asarray(op['weights']))))
            cv_rows.append(dict(time_s=t1,duration_s=1800.,model_meter_mean_m3s=integrals['meter']/1800,model_gate_mean_m3s=integrals['gate']/1800,
                 offtake_mean_m3s=integrals['take']/1800,native_storage_rate_m3s=integrals['cv_storage']/1800,
                 continuity_residual_m3s=(integrals['meter']-integrals['gate']-integrals['take']-integrals['cv_storage'])/1800))
            v=volume(m)
            network_rows.append(dict(start_s=t0,end_s=t1,net_boundary_input_m3=integrals['global_flux'],
                 discrete_continuity_storage_change_m3=integrals['global_storage'],discrete_balance_residual_m3=integrals['global_storage']-integrals['global_flux'],
                 geometric_storage_change_m3=v-geometric_volume,geometry_minus_discrete_change_m3=v-geometric_volume-integrals['global_storage'],
                 maximum_step_discrete_residual_m3s=integrals['global_max_residual'],accepted_steps=int(integrals['steps']),
                 volume_scope='内部连接A*L及非定水位主节点体积；原端长/内部附加储水均0；原截高及闭口槽缝未修复'))
            geometric_volume=v;domain_rows.append(dict(time_s=t1,**contour_values(m,heights,open_links,endpoint_arrays)))
            status.update(completed_cycles=cycle+1,completed_successful_substeps=success,rejected_attempts=rejected,retries=retries,
                 model_time_s=t1,maximum_control_volume_residual_m3s=maxcv,maximum_nonboundary_node_residual_m3s=maxnode,
                 maximum_internal_residual_m3s=maxinternal,maximum_full_network_discrete_residual_m3s=maxnetwork,
                 control_volume_exceeded_step_count=cv_bad,nonboundary_node_exceeded_step_count=node_bad,
                 iteration_counts=iterations,wall_seconds=time.time()-started)
            if (cycle+1)%48==0 or cycle==1439:
                assert all(arrsha(getattr(m,key))==value for key,value in fixed.items())
                assert all(arrsha(getattr(m,key))==value for key,value in internal_geometry.items())
                persist();dump(out/'进度.json',status);print(json.dumps(status),flush=True)
        assert retries==rejected and len(node_rows)==84960 and len(mapped_rows)==61920 and len(meter_rows)==63360
        assert len(network_rows)==len(cv_rows)==1440
        np.savez_compressed(out/'末态.npz',**{key:getattr(m,key) for key in STATE_KEYS})
        # 参考分支沿用41/10已经完成且精确复现06的整月，不再次推进。
        assert all(sha(p)==h for p,h in sources.items())
        status.update(status='COMPLETED',full_30day_completed=True,flow_operator_count=44,
             all_successful_steps_control_volume_gate_pass=cv_bad==0,all_successful_steps_all_nonboundary_nodes_under_1e6=node_bad==0,
             parameters_unchanged=True,internal_geometry_unchanged=True,source_hashes_unchanged=True,
             endpoint_hook_applied=False,gate_bottom_hook_applied=False,reference_reused_without_rerun=True,
             assimilation=False,baseline_replaced=False,engineering_acceptance=False,runtime_identity=runtime_identity(),wall_seconds=time.time()-started)
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),full_30day_completed=False,
             completed_successful_substeps=success,rejected_attempts=rejected,model_time_s=float(m.t) if m is not None else 0.,wall_seconds=time.time()-started)
        persist()
        if m is not None:np.savez_compressed(out/'停止状态.npz',**{key:getattr(m,key) for key in STATE_KEYS},time_s=np.array(m.t))
        raise
    finally:
        failure_handle.close();dump(out/'结果.json',status);dump(out/'进度.json',status)
        if shared is not None:shutil.rmtree(shared.CACHE,ignore_errors=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['q22','q26'])
    run_branch(parser.parse_args().case)
