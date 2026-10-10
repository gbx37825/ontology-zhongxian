"""西赵河源资料连接修复的隔离验证；不改n/C、不使用内部观测更新。"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import time
import traceback
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R41 = OUT.parent
R = R41.parent
spec = importlib.util.spec_from_file_location('full38_west41', R/'38_测点假设下的联合辨识/06_全线联合参数转移/全线固定参数转移.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
api = full.api
B = api.FROZEN
REFERENCE = R/'38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数'
PATCH = R41/'03_结构转换核查/西赵河最小候选字段差分.json'
UP = 'POOL4-1-JJZ4DHX1'
DOWN = 'POOL4-2-DHX1JJZ5'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def engine_for(source):
    def original_build(*args, gate_max_normalization=None, **kwargs):
        norms = {k: float(v.get('b', 1) or 1) for k, v in source['baseData']['gates'].items()}
        norms.update(gate_max_normalization or {})
        c = api.parse_wxq_json(source, gate_max_openings=norms, section_geometry_mode='safe', minimum_initial_depth_m=7.)
        c['superjunctions'].loc[:, 'h_0'] = 7.
        c['superjunctions'].loc[:, 'max_depth'] = np.maximum(c['superjunctions'].max_depth.to_numpy(), 7.)
        c['orifices'].loc[:, 'C'] = c['orifices'].name.map(api.data()[3]).fillna(c['orifices'].C)
        e = api.PipedreamHydraulicEngine(c, dt=15., backend='numba')
        e.model._method = 'lsq'
        return e, source, c
    old = api._physics._legacy_build
    try:
        api._physics._legacy_build = original_build
        e, s, c = api._physics.build_engine_physics(gate_max_normalization=api.data()[2])
    finally:
        api._physics._legacy_build = old
    return e, s, c


def make_candidate(source):
    raw = copy.deepcopy(source)
    diff = json.loads(PATCH.read_text())
    assert diff['base_sha256'] == sha(B/'共享输入/C07.json')
    for row in diff['patches']:
        parts = row['path'].strip('/').split('/')
        item = raw
        for part in parts[:-1]:
            item = item[int(part)] if isinstance(item, list) else item[part]
        key = int(parts[-1]) if isinstance(item, list) else parts[-1]
        if row['op'] == 'replace':
            assert item[key] == row['old']
            item[key] = copy.deepcopy(row['value'])
        elif isinstance(item, list):
            item.insert(key, copy.deepcopy(row['value']))
        else:
            assert key not in item
            item[key] = copy.deepcopy(row['value'])
    return raw


def link_centres(c, mesh):
    rows = c['links'][c['links'].k == mesh['k']]
    xs = dict(zip(mesh['ids'], mesh['x']))
    pairs = sorted(((xs[row.j_0]+xs[row.j_1])/2., int(row.id)) for row in rows.itertuples())
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs], int)


def project(old_e, old_s, old_c, e, source, c):
    old, new = old_e.model, e.model
    assert old_e.node_names == e.node_names and old_e.gate_names == e.gate_names
    assert old_c['superlinks'].name.tolist() == c['superlinks'].name.tolist()
    for key in ['H_j', 'Q_o', 'Q_uk', 'Q_dk']:
        getattr(new, key)[:] = getattr(old, key)
    before = api.meshes(old_s, old_c)
    after = api.meshes(source, c)
    log = []
    filled = np.zeros(new.NIk, bool)
    for name, meta in c['channel_discretization'].items():
        prev = old_c['channel_discretization'][name]
        if name not in after:
            oi = np.asarray([p['junction_id'] for p in prev['sections']], int)
            ni = np.asarray([p['junction_id'] for p in meta['sections']], int)
            assert len(oi) == len(ni) and np.array_equal(old._z_inv_Ik[oi], new._z_inv_Ik[ni])
            new.h_Ik[ni] = old.h_Ik[oi]
            old_links = np.flatnonzero(old._ki == prev['superlink_id'])
            new_links = np.flatnonzero(new._ki == meta['superlink_id'])
            assert len(old_links) == len(new_links)
            new.Q_ik[new_links] = old.Q_ik[old_links]
        else:
            om, nm = before[name], after[name]
            assert om['sign'] == nm['sign']
            oi, ni = om['ids'], nm['ids']
            by_x = dict(zip(om['x'], oi))
            for x, new_id in zip(nm['x'], ni):
                # 两个发生物理身份变化的端点由不变的管口共享节点给定，禁止同名复用或外推。
                node = 'NODE-pool4-1' if name == UP and x == 69605. else ('NODE-pool4-2' if name == DOWN and x == 69789. else None)
                if node:
                    head = old.H_j[old_e.node_index[node]]
                    new.h_Ik[new_id] = head-new._z_inv_Ik[new_id]
                    log.append(dict(channel=name, x=float(x), node=node, absolute_head_m=float(head), new_depth_m=float(new.h_Ik[new_id])))
                else:
                    assert x in by_x, (name, x)
                    old_id = by_x[x]
                    assert old._z_inv_Ik[old_id] == new._z_inv_Ik[new_id]
                    new.h_Ik[new_id] = old.h_Ik[old_id]
            ox, oq = link_centres(old_c, om)
            nx, nq = link_centres(c, nm)
            by_centre = dict(zip(ox, oq))
            for x, new_id in zip(nx, nq):
                if x in by_centre:
                    new.Q_ik[new_id] = old.Q_ik[by_centre[x]]
                else:
                    # 修复的两个端部区间没有新增分水，采用同一原管口通量衔接。
                    assert name in [UP, DOWN]
                    k = int(om['k'])
                    physical = (old.Q_dk[k] if om['sign'] == 1 else old.Q_uk[k])*om['sign'] if name == UP else (old.Q_uk[k] if om['sign'] == 1 else old.Q_dk[k])*om['sign']
                    new.Q_ik[new_id] = physical/nm['sign']
                    log.append(dict(channel=name, flow_interval_centre_m=float(x), physical_initial_flow_m3s=float(physical)))
        filled[ni] = True
    assert filled.all() and np.isfinite(new.h_Ik).all() and (new.h_Ik >= 0).all()
    new.t = 0.
    new.iter_count = 0
    return log


def observations(old_s, old_c, source, c, model):
    operators = json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text())
    before, after = api.meshes(old_s, old_c), api.meshes(source, c)
    mapped = []
    for op in operators:
        assert op['state'] == 'h_Ik'
        names = [name for name, mesh in before.items() if set(op['indices']).issubset(set(mesh['ids']))]
        assert len(names) == 1
        name = names[0]
        ids, weights = api.weights_at(after[name], op['chainage'])
        mapped.append({**op, 'channel': name, 'indices': ids.tolist(), 'weights': weights.tolist(),
                       'offset': float(model._z_inv_Ik[ids]@weights)})
    return mapped


def input_node_keys(source, converted):
    meshes = api.meshes(source, converted)
    keys = {}
    for name, meta in converted['channel_discretization'].items():
        if name in meshes:
            points = zip(meshes[name]['ids'], meshes[name]['x'])
        else:
            points = [(p['junction_id'], p['distance']) for p in meta['sections']]
        for i, x in points:
            key = (name, float(x))
            assert key not in keys
            keys[key] = int(i)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['reference_preflight', 'reference_full', 'western_repair'])
    args = parser.parse_args()
    out = OUT/args.case
    out.mkdir(exist_ok=False)
    begin = time.time()
    status = dict(status='RUNNING', case=args.case, completed_cycles=0, pid=os.getpid())
    rows=[]; mapped=[]; ranges=[]; meter_rows=[]; water_rows=[]
    try:
        original = copy.deepcopy(api.data()[0])
        oe, osrc, oc = engine_for(original)
        with np.load(B/'共享输入/初始化后状态_0s.npz') as z:
            for key in api.STATE_KEYS:
                getattr(oe.model, key)[:] = z[key]
        oe.model.t = 0.; oe.model.iter_count = 0
        projection = []
        if args.case == 'western_repair':
            source = make_candidate(original)
            e, source, c = engine_for(source)
            projection = project(oe, osrc, oc, e, source, c)
        else:
            e, source, c = oe, osrc, oc
        m = e.model
        assert np.count_nonzero(m._dx_uk)==np.count_nonzero(m._dx_dk)==np.count_nonzero(m._A_SIk)==0
        assert m.n_w==m.n_p==0
        f = api.Forcing(e, source, c)
        ops = observations(osrc, oc, source, c, m)
        save(out/'实际位置观测算子.json', ops)
        old_f = api.Forcing(oe, osrc, oc)
        assert all(np.array_equal(getattr(f, k), getattr(old_f, k), equal_nan=True) for k in ['times', 'h', 'q', 'u'])
        keys0,keys1=input_node_keys(osrc,oc),input_node_keys(source,c)
        assert len(keys0)==oe.model.NIk and len(keys1)==m.NIk
        shared=keys0.keys() & keys1.keys()
        for key in shared:assert np.array_equal(old_f.q0[:,keys0[key]],f.q0[:,keys1[key]])
        for key in keys0.keys()-keys1.keys():assert np.count_nonzero(old_f.q0[:,keys0[key]])==0
        for key in keys1.keys()-keys0.keys():assert np.count_nonzero(f.q0[:,keys1[key]])==0
        q0_roundoff=float(np.max(np.abs(f.q0.sum(axis=1)-old_f.q0.sum(axis=1))))
        assert q0_roundoff < 1e-10
        save(out/'分水逐物理节点一致性.json',dict(shared_nodes=len(shared),all_shared_values_exact=True,
             removed_nodes_all_zero=True,added_nodes_all_zero=True,sum_roundoff_max_m3s=q0_roundoff))
        assert all('按模型真实桩号' in row['mode'] for row in f.routing)
        flow_api = api._bind_definitions(api.DEPS/'水力运行脚本/试验公共.py',
            ['flow_operator','flow_operators'],dict(np=np,pd=pd,re=__import__('re'),R=R,meshes=api.meshes), '_41_frozen_flow_operators')
        flow_ops, flow_audit = flow_api.flow_operators(e,source,c)
        save(out/'测流位置算子与适用范围.json',dict(operators=flow_ops,audit=flow_audit,
             qualification='沿用冻结测点位置；端部取最近连接中心的算子属于近似，不宣称全部为严格同坐标读取。'))
        params = {k: arrsha(getattr(m,k)) for k in full.PARAM_KEYS}
        assert np.array_equal(m._Co, oe.model._Co)
        for name in c['superlinks'].name:
            k = int(c['superlinks'].loc[c['superlinks'].name == name, 'id'].iloc[0])
            assert np.unique(m._n_ik[m._ki == k]).tolist() == np.unique(oe.model._n_ik[oe.model._ki == k]).tolist()
        sources = {str(p):sha(p) for p in [Path(__file__), Path(full.__file__),Path(api.__file__),PATCH, R41/'修复与验证约定.json', B/'共享输入/C07.json', B/'共享输入/初始化后状态_0s.npz', B/'共享输入/作者已核实修正观测层.csv',B/'共享输入/2025开度归一化上限.json',B/'共享输入/Pipedream闸流系数.json',B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json',R/'02_输入/测点位置映射.csv']}
        sources.update({x['file']:x['sha256'] for x in api.runtime_identity()['modules']})
        sources.update({str(REFERENCE/name):sha(REFERENCE/name) for name in ['运行配置.json','半小时水位流量.csv','末态.npz']})
        def volume(mm):
            trial = copy.deepcopy(mm);trial.link_hydraulic_geometry();trial.compute_storage_volumes()
            return float(np.sum(trial._A_ik*trial._dx_ik)+np.sum(trial.V_j[~trial.bc]))
        modified = []
        for name in [UP, DOWN]:
            mesh = api.meshes(source,c)[name]
            for x, i in zip(mesh['x'],mesh['ids']):
                if 69521. <= x <= 69872.:
                    section = next(p for p in c['channel_discretization'][name]['sections'] if p['junction_id']==i)
                    modified.append((name,float(x),int(i),float(section['height'])))
        cfg=dict(case=args.case,source_hashes=sources,runtime_identity=api.runtime_identity(),projection=projection,
                 original_initial_volume_m3=volume(oe.model),candidate_initial_volume_m3=volume(m),
                 geometry_changed=args.case=='western_repair',parameters_fixed=True,internal_observation_updates=0,
                 initial_six_states_identical=args.case!='western_repair',modified_open_points=modified,
                 initial_cache_rule='与原G00一致：构模缓存后恢复六状态；初始几何体积仅在副本刷新计量，未回灌',
                 reference=str(REFERENCE),baseline_replaced=False)
        cfg['fixed_parameter_sha256']=params
        cfg['forcing_array_sha256']={k:arrsha(getattr(f,k)) for k in ['times','h','q','q0','u']}
        cfg['initial_state_sha256']={k:arrsha(getattr(m,k)) for k in api.STATE_KEYS}
        if args.case!='western_repair':
            reference_cfg=json.loads((REFERENCE/'运行配置.json').read_text())
            assert cfg['fixed_parameter_sha256']==reference_cfg['parameter_sha256']
            assert cfg['forcing_array_sha256']==reference_cfg['forcing_array_sha256']
            assert cfg['initial_state_sha256']==reference_cfg['initial_state_sha256']
        save(out/'运行配置.json',cfg)
        groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
        retries=0
        def check_range(t):
            vals=[dict(time_s=float(t),channel=name,chainage_m=x,head_m=float(m.h_Ik[i]+m._z_inv_Ik[i]),input_top_m=float(height+m._z_inv_Ik[i]),excess_m=float(m.h_Ik[i]-height)) for name,x,i,height in modified]
            if args.case=='western_repair' and any(row['excess_m']>1e-7 for row in vals):
                save(out/'修复区输入轮廓门槛停止.json',dict(status='STOPPED_AT_PREDECLARED_GEOMETRY_GATE',points=vals))
                raise ValueError('修复区水位超输入轮廓，候选顶部假设不满足；不抬高、不夹水位。')
            return vals
        check_range(0.)
        cycles=48 if args.case=='reference_preflight' else 1440
        geometric_volume=cfg['candidate_initial_volume_m3']
        for cycle in range(cycles):
            flux_volume=discrete_volume=0.; max_step_residual=0.; accepted_steps=0
            for ts in np.arange(cycle*1800.,(cycle+1)*1800.,15.):
                pending=[(float(ts),15.)]
                while pending:
                    t,dt=pending.pop(0);kw=f.at(t);kw.update(dt=dt,num_iter=20,strict=True)
                    try:m.step(**kw)
                    except RuntimeError as exc:
                        if 'did not converge' not in str(exc) or dt<=1.875:raise
                        retries+=1;pending[:0]=[(t,dt/2),(t+dt/2,dt/2)];continue
                    assert all(np.isfinite(getattr(m,k)).all() for k in api.STATE_KEYS)
                    check_range(m.t)
                    # 全网控制体不包含定水位边界节点；含跨边界渠段/闸孔通量和全部内部取水。
                    external=float(np.sum(kw['Q_in'][~m.bc])+np.sum(kw['Q_0Ik']))
                    external+=float(np.sum(m.Q_uk[m.bc[m._J_uk]])-np.sum(m.Q_dk[m.bc[m._J_dk]]))
                    external+=float(np.sum(m.Q_o[m.bc[m._J_uo]])-np.sum(m.Q_o[m.bc[m._J_do]]))
                    storage=float(np.sum(m._E_Ik*(m.h_Ik-m.states['h_Ik']))*dt+
                                  np.sum(m._A_sj[~m.bc]*(m.H_j[~m.bc]-m.states['H_j'][~m.bc])))
                    flux_volume+=external*dt; discrete_volume+=storage
                    max_step_residual=max(max_step_residual,abs(storage/dt-external));accepted_steps+=1
            t=(cycle+1)*1800
            for station,(ju,jd,ids) in groups.items():rows.append(dict(time_s=t,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
            for op in ops:mapped.append(dict(time_s=t,station=op['station'],metric=op['metric'],mapped_sim=float(m.h_Ik[op['indices']]@op['weights']+op['offset']),group='旧参与同化测点' if op['assimilated'] else '旧留出检验测点',chainage_m=op['chainage']))
            for op in flow_ops:meter_rows.append(dict(time_s=t,station=op['station'],chainage_m=op['chainage'],model_flow_m3s=float(getattr(m,op['state'])[op['indices']]@op['weights'])))
            v=volume(m)
            water_rows.append(dict(start_s=cycle*1800,end_s=t,net_boundary_input_m3=flux_volume,
                discrete_continuity_storage_change_m3=discrete_volume,discrete_balance_residual_m3=discrete_volume-flux_volume,
                geometric_storage_change_m3=v-geometric_volume,geometry_minus_discrete_change_m3=v-geometric_volume-discrete_volume,
                maximum_step_discrete_residual_m3s=max_step_residual,accepted_steps=accepted_steps,
                volume_scope='内部连接A*L及非定水位主节点体积；原端段长度为0；原输入轮廓截高和有压槽缝效应另行解释'))
            geometric_volume=v
            ranges.extend(check_range(t));status['completed_cycles']=cycle+1
            if (cycle+1)%48==0:
                assert all(arrsha(getattr(m,k))==v for k,v in params.items())
                status.update(days=(cycle+1)/48,wall_seconds=time.time()-begin,retries=retries);save(out/'进度.json',status);print(json.dumps(status),flush=True)
        pd.DataFrame(rows).to_csv(out/'半小时水位流量.csv',index=False)
        pd.DataFrame(mapped).to_csv(out/'实际位置水位.csv',index=False)
        pd.DataFrame(ranges).to_csv(out/'修复区半小时轮廓范围.csv',index=False)
        pd.DataFrame(meter_rows).to_csv(out/'测流位置算子流量.csv',index=False)
        pd.DataFrame(water_rows).to_csv(out/'全网半小时水量账.csv',index=False)
        np.savez_compressed(out/'末态.npz',**{k:getattr(m,k) for k in api.STATE_KEYS})
        if args.case!='western_repair':
            old=pd.read_csv(REFERENCE/'半小时水位流量.csv');old=old[old.time_s<=cycles*1800].reset_index(drop=True)
            new=pd.read_csv(out/'半小时水位流量.csv')
            exact=new.equals(old);save(out/'原基线逐值复现.json',dict(exact_node_trajectory_equal=exact,rows=len(new),coverage_start_s=0,coverage_end_s=cycles*1800))
            assert exact, '新验证入口未逐值复现原G00'
            if args.case=='reference_full':
                with np.load(REFERENCE/'末态.npz') as z:assert all(np.array_equal(getattr(m,k),z[k]) for k in api.STATE_KEYS)
        assert all(sha(p)==v for p,v in sources.items())
        status.update(status='COMPLETED',wall_seconds=time.time()-begin,retries=retries,baseline_replaced=False)
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-begin)
        for name,values in [('未完成半小时水位流量',rows),('未完成实际位置水位',mapped),('未完成轮廓范围',ranges),('未完成测流位置算子流量',meter_rows),('未完成全网半小时水量账',water_rows)]:
            if values:pd.DataFrame(values).to_csv(out/(name+'.csv'),index=False)
        if 'm' in locals():np.savez_compressed(out/'停止时状态.npz',**{k:getattr(m,k) for k in api.STATE_KEYS})
        raise
    finally:save(out/'进度.json',status)


if __name__=='__main__':main()
