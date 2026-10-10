"""10配对入口：共同S03数值精度下，仅endpoint安装已审核09端面几何hook。

不修改06，不修改西赵河连接，不安装S02。首次执行须经主进程静态审查。
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

HERE=Path(__file__).resolve().parent
R41=HERE.parent
R=R41.parent
B=R/'26_冻结基线与后续归因'
DEPS=B/'共享依赖'
REFERENCE06=R41/'06_闸流水量一致性修复/G00_更严闸流一致性'
RUNNER06=R41/'06_闸流水量一致性修复/原参数三十天闸流一致性复算.py'
HELPER01=R41/'01_求解器核查/面积储水与闸流最小复现.py'
HOOK09=R41/'09_端部几何一致性核查/端部独立几何候选.py'
HOOK_SHA='886e6fbbe9c95fb2c0ca6b5778aa78fe491c8f95969a073ac3437e590805e42f'
STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
PARAM_KEYS=('_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk')
INTERNAL_GEOMETRY_KEYS=('_g1_ik','_g2_ik','_g3_ik','_g4_ik','_g5_ik','_g6_ik','_g7_ik',
                        '_dx_ik','_dx_uk','_dx_dk','_z_inv_Ik','_z_inv_j')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


# 06模块顶层仅定义入口/常量，main不会被执行；复用已有保存与运行身份核验。
common06=load(RUNNER06,'_41_endpoint_common06')
dump=common06.dump
runtime_identity=common06.runtime_identity


def build_runtime():
    shared=load(HELPER01,'_41_endpoint_build_source')
    common=shared.bind(DEPS/'水力运行脚本/试验公共.py',
         ['Forcing','meshes','weights_at','flow_operator','flow_operators','hydraulic_volume'],
         dict(np=np,pd=pd,re=shared.re,Path=Path,R=R),'_41_endpoint_frozen_common')
    physics=shared.bind(DEPS/'并联结构物理修复.py',
         ['_correct_section','_link_geometry','_up_geometry','_down_geometry','_link_coeffs','attach_parallel_physics','build_engine_physics'],
         dict(np=np,types=shared.types,copy=copy,hashlib=hashlib,Path=Path,R=R,
              PipedreamHydraulicEngine=shared.PipedreamHydraulicEngine,_legacy_build=shared.legacy_full_build),
         '_41_endpoint_frozen_physics')
    advance=shared.bind(DEPS/'参数并行运行脚本/常驻水力进程.py',['advance'],
                        dict(np=np,STATE_KEYS=STATE_KEYS),'_41_endpoint_original_advance').advance
    restore=shared.bind(DEPS/'参数并行运行脚本/局地参数EnKF.py',['restore'],{},'_41_endpoint_restore').restore
    e,s,c=physics.build_engine_physics(gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()))
    return shared,common,advance,restore,e,s,c


def contour_metadata(m,source,c):
    heights=np.full(m.NIk,np.inf)
    endpoints=[];channels=[]
    for name in source['baseData']['channels']:
        meta=c['channel_discretization'][name];k=int(meta['superlink_id']);channels.append(k)
        for point in meta['sections']:heights[point['junction_id']]=point['height']
        for side,point,internal,supernode,link in [
                ('u',meta['sections'][0],m._I_1k[k],m._J_uk[k],m._i_1k[k]),
                ('d',meta['sections'][-1],m._I_Np1k[k],m._J_dk[k],m._i_nk[k])]:
            assert int(point['junction_id'])==int(internal)
            endpoints.append(dict(channel=name,side=side,superlink_id=k,internal_node=int(internal),
                 superjunction=int(supernode),adjacent_internal_link=int(link),flow_sign=int(meta['flow_sign']),
                 original_adjacent_g1_m=float(m._g1_ik[link]),endpoint_g1_m=float(point['height']),
                 endpoint_bottom_width_m=float(point['bottom_width']),endpoint_side_slope=float(point['side_slope']),
                 endpoint_bed_m=float(point['z_inv']),source_section=point['name']))
    arrays={key:np.asarray([row[key] for row in endpoints]) for key in
            ['superlink_id','internal_node','superjunction','original_adjacent_g1_m','endpoint_g1_m']}
    arrays['upstream']=np.asarray([row['side']=='u' for row in endpoints])
    return heights,np.flatnonzero(np.isin(m._ki,channels)),endpoints,arrays


def contour_values(m,heights,open_links,ends):
    # 统一用接受态评估固定轮廓；只读统计，不决定模型用旧/新几何。
    node_ids=np.flatnonzero(np.isfinite(heights))
    node_excess=m.h_Ik[node_ids]-heights[node_ids]
    y=(m.h_Ik[m._Ik[open_links]]+m.h_Ik[m._Ip1k[open_links]])/2
    link_excess=y-m._g1_ik[open_links]
    k,i,j=ends['superlink_id'],ends['internal_node'],ends['superjunction']
    bed=np.where(ends['upstream'],m._z_inv_uk[k],m._z_inv_dk[k])
    theta=np.where(ends['upstream'],m._theta_uk[k],m._theta_dk[k])
    node_depth=m.H_j[j]-bed;effective=theta*node_depth;internal=m.h_Ik[i]
    avg=(internal+effective)/2
    old=avg-ends['original_adjacent_g1_m'];source_mean=avg-ends['endpoint_g1_m']
    source_internal=internal-ends['endpoint_g1_m'];source_super=node_depth-ends['endpoint_g1_m']
    result={}
    for name,values in [('original_open_node',node_excess),('original_internal_link',link_excess),
             ('endpoint_mean_vs_original_adjacent',old),('endpoint_mean_vs_source',source_mean),
             ('endpoint_internal_vs_source',source_internal),('endpoint_superjunction_vs_source',source_super)]:
        values=np.asarray(values)
        result[name+'_above_count']=int((values>1e-9).sum())
        result[name+'_max_excess_m']=float(values.max())
    upper=np.minimum(ends['original_adjacent_g1_m'],ends['endpoint_g1_m'])
    outside=(np.maximum(internal,effective)>upper+1e-9)|(np.minimum(internal,effective)<-1e-9)
    result['endpoint_outside_either_contour_count']=int(outside.sum())
    raw_outside=(np.maximum(internal,node_depth)>upper+1e-9)|(np.minimum(internal,node_depth)<-1e-9)
    result['endpoint_raw_depth_outside_either_contour_count']=int(raw_outside.sum())
    result['endpoint_raw_depth_below_zero_count']=int((np.minimum(internal,node_depth)<-1e-9).sum())
    return result


def residuals(m,before,kw,dt,W,jids,iids,wj,wi,qids,qw,gids,node_ids):
    dH,dh=m.H_j-before['H_j'],m.h_Ik-before['h_Ik']
    cv_storage=float(wj@(m._A_sj[jids]*dH[jids])+wi@(m._E_Ik[iids]*dh[iids]*dt))
    take=-float(wj@kw['Q_in'][jids]+wi@kw['Q_0Ik'][iids])*dt
    qm,qg=float(m.Q_ik[qids]@qw),float(m.Q_o[gids].sum())
    cv=qm-qg-take/dt-cv_storage/dt
    rj,ri=kw['Q_in'].copy(),kw['Q_0Ik'].copy()
    for vector,ids,values in [(rj,m._J_uk,-m.Q_uk),(rj,m._J_dk,m.Q_dk),
            (rj,m._J_uo,-m.Q_o),(rj,m._J_do,m.Q_o),(ri,m._I_1k,m.Q_uk),
            (ri,m._I_Np1k,-m.Q_dk),(ri,m._Ik,-m.Q_ik),(ri,m._Ip1k,m.Q_ik)]:np.add.at(vector,ids,values)
    rj-=m._A_sj*dH/dt;ri-=m._E_Ik*dh
    assert abs(cv-float(W[:m.M]@rj+W[m.M:]@ri))<1e-9
    # 与05相同：全网控制体不含定水位主节点，端长及内部附加储水已断言为0。
    external=float(np.sum(kw['Q_in'][~m.bc])+np.sum(kw['Q_0Ik']))
    external+=float(np.sum(m.Q_uk[m.bc[m._J_uk]])-np.sum(m.Q_dk[m.bc[m._J_dk]]))
    external+=float(np.sum(m.Q_o[m.bc[m._J_uo]])-np.sum(m.Q_o[m.bc[m._J_do]]))
    storage=float(np.sum(m._E_Ik*dh)*dt+np.sum(m._A_sj[~m.bc]*dH[~m.bc]))
    row=dict(Q16_cut_m3s=qm,Q16_gate_m3s=qg,offtake_volume_m3=take,native_storage_change_m3=cv_storage,
         continuity_residual_m3s=cv,max_nonboundary_node_residual_m3s=float(abs(rj[node_ids]).max()),
         max_internal_residual_m3s=float(abs(ri).max()),net_boundary_input_m3=external*dt,
         full_network_discrete_storage_change_m3=storage,full_network_discrete_residual_m3=storage-external*dt,
         full_network_discrete_residual_m3s=storage/dt-external)
    return row,rj[node_ids].copy()


def run_branch(case):
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
        assert sha(HOOK09)==HOOK_SHA
        zero_check=json.loads((R41/'09_端部几何一致性核查/端部候选零步核验结果.json').read_text())
        assert zero_check['status']=='PASSED'
        oldcfg=json.loads((REFERENCE06/'运行配置.json').read_text());oldresult=json.loads((REFERENCE06/'结果.json').read_text())
        assert oldresult['status']=='COMPLETED' and oldresult['full_30day_completed']
        sources={**oldcfg['source_hashes'],**protocol['source_sha256'],str(protocol_path):sha(protocol_path),
                 str(REFERENCE06/'运行配置.json'):sha(REFERENCE06/'运行配置.json')}
        assert all(sha(p)==h for p,h in sources.items())
        shared,common,advance,restore,e,source,c=build_runtime();m=e.model
        with np.load(B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{key:z[key].copy() for key in STATE_KEYS})
        m.t=0.;m.iter_count=0
        initial={key:arrsha(getattr(m,key)) for key in STATE_KEYS}
        fixed={key:arrsha(getattr(m,key)) for key in PARAM_KEYS}
        internal_geometry={key:arrsha(getattr(m,key)) for key in INTERNAL_GEOMETRY_KEYS}
        assert initial==oldcfg['initial_state_sha256'] and fixed==oldcfg['parameter_sha256']
        assert np.count_nonzero(m._dx_uk)==np.count_nonzero(m._dx_dk)==np.count_nonzero(m._A_SIk)==0 and m.n_w==m.n_p==0
        manifest=None
        if case=='endpoint':
            hook=load(HOOK09,'_41_endpoint_geometry_hook')
            manifest=hook.attach_endpoint_geometry(m,c,source)
        assert {key:arrsha(getattr(m,key)) for key in STATE_KEYS}==initial
        assert all(arrsha(getattr(m,key))==value for key,value in fixed.items())
        assert all(arrsha(getattr(m,key))==value for key,value in internal_geometry.items())
        f=common.Forcing(e,source,c);forcing_hashes={key:arrsha(getattr(f,key)) for key in ['times','h','q','q0','u']}
        assert forcing_hashes==oldcfg['forcing_array_sha256']
        ops=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());assert len(ops)==43
        flow_ops,flow_audit=common.flow_operators(e,source,c);assert len(flow_ops)==44
        dump(out/'实际位置观测算子.json',ops)
        dump(out/'测流位置算子与适用范围.json',dict(operators=flow_ops,audit=flow_audit,
             qualification='与05同一冻结测点位置口径；端部最近连接中心近似保留，不宣称44点均为严格同坐标仪器真值。'))
        groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int))
                for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};assert len(groups)==59
        qop=next(o for o in flow_ops if o['station']=='JZZ16');assert qop==oldcfg['flow_operator']
        qids,qw=np.asarray(qop['indices']),np.asarray(qop['weights'])
        g00=json.loads((R/'38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数/运行配置.json').read_text())
        gids=np.asarray(g00['C_indices'])
        W=np.load(R/'30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy')
        jids,iids=np.flatnonzero(W[:m.M]),np.flatnonzero(W[m.M:]);wj,wi=W[jids],W[m.M+iids]
        assert arrsha(W)==oldcfg['control_volume_weights_sha256']
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
        identity=runtime_identity();assert not identity['numba_jit_disabled'] and identity['momentum_compiled_signatures']>0
        dump(out/'运行配置.json',dict(case=case,source_hashes=sources,protocol=protocol,
             initial_state_sha256=initial,parameter_sha256=fixed,forcing_array_sha256=forcing_hashes,
             internal_geometry_sha256=internal_geometry,initial_geometric_volume_m3=initial_volume,
             volume_scope='内部连接A*L及非定水位主节点体积；在副本刷新几何计量，不回灌',
             endpoint_geometry_manifest=manifest,endpoint_hook_applied=case=='endpoint',reference_06=str(REFERENCE06),
             assimilation=False,state_corrections=0,parameters_fixed=True,six_initial_states_match_06=True,
             source_geometry_changed=False,internal_midpoint_geometry_changed=False,
             runtime_identity=identity,baseline_replaced=False))
        np.savez_compressed(out/'实际初始化_0s.npz',**{key:getattr(m,key) for key in STATE_KEYS})
        initial_domain=contour_values(m,heights,open_links,endpoint_arrays)
        expected_domain_count=next(row['open_domain_counts_preserved'] for row in zero_check['results'] if row['state']=='固定初态0s')
        assert initial_domain['endpoint_outside_either_contour_count']==expected_domain_count==78
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
        exact=None
        if case=='reference':
            exact={name:pd.read_csv(out/name).equals(pd.read_csv(REFERENCE06/name)) for name in ['半小时水位流量.csv','实际位置水位.csv']}
            with np.load(REFERENCE06/'末态.npz') as z:exact['six_final_states']=all(np.array_equal(getattr(m,key),z[key]) for key in STATE_KEYS)
            dump(out/'与06逐值复现核验.json',dict(checks=exact,all_exact=all(exact.values()),reference=str(REFERENCE06),coverage_s=[0,2592000]))
            assert all(exact.values()),'新增参考分支未逐值复现06，不得比较端面候选精度'
        assert all(sha(p)==h for p,h in sources.items())
        status.update(status='COMPLETED',full_30day_completed=True,flow_operator_count=44,
             all_successful_steps_control_volume_gate_pass=cv_bad==0,all_successful_steps_all_nonboundary_nodes_under_1e6=node_bad==0,
             parameters_unchanged=True,internal_geometry_unchanged=True,source_hashes_unchanged=True,
             endpoint_hook_applied=case=='endpoint',reference_06_exact_checks=exact,
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
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['reference','endpoint'])
    run_branch(parser.parse_args().case)
