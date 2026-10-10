"""冻结0s/30天状态的端部几何代数核查；不调用step、不修改任何冻结文件。"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ['NUMBA_CACHE_DIR'] = '/private/tmp/pipedream_frozen_41_numba'
HERE = Path(__file__).resolve().parent
R41 = HERE.parent
ROOT = R41.parent
FROZEN = ROOT / '26_冻结基线与后续归因'
DEPS = FROZEN / '共享依赖'
spec = importlib.util.spec_from_file_location('structure41_endpoint_audit', R41/'03_结构转换核查/结构转换逐项核查.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
import numpy as np
import pandas as pd
from pipedream_solver.nsuperlink import numba_boundary_geometry, gamma_uk, gamma_dk

SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARRSHA = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
STATE_KEYS = ('H_j', 'h_Ik', 'Q_ik', 'Q_uk', 'Q_dk', 'Q_o')
CODE = {'rect_closed': 2, 'rect_open': 3, 'trapezoidal': 5}
GEOMETRY_METRICS = ('A', 'Pe', 'R', 'B')


def write(name, data):
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def endpoint_inputs(converted, model, side):
    points = [None] * model.NK
    shapes = [None] * model.NK
    names = [None] * model.NK
    channel_names = set(source['baseData']['channels'])
    for name, meta in converted['channel_discretization'].items():
        k = meta['superlink_id']
        p = meta['sections'][0 if side == 'u' else -1]
        points[k] = p
        names[k] = name
        shapes[k] = p['shape_kind'] if name in channel_names else str(converted['superlinks'].loc[k, 'shape'])
        assert shapes[k] in CODE, (name, shapes[k])
    g = [np.array([p[field] for p in points], dtype=float) for field in ['height','bottom_width','side_slope']]
    return points, names, shapes, g, np.asarray([CODE[s] for s in shapes], np.int64)


def compare(model, converted, label, state_path):
    with np.load(state_path) as z:
        for key in STATE_KEYS:
            getattr(model,key)[:] = z[key]
    before = {key: ARRSHA(getattr(model,key)) for key in STATE_KEYS + ('_n_ik','_Co','_C_uk','_C_dk','_dx_ik','_dx_uk','_dx_dk')}
    assert not np.any(model._dx_uk) and not np.any(model._dx_dk)
    assert model._bc_method == 'z'
    # 这些存档没有干端点；theta必须与当前绝对水头判据一致，不人为改动深度规则。
    assert np.array_equal(model._theta_uk, (model.H_j[model._J_uk] >= model._z_inv_uk).astype(float))
    assert np.array_equal(model._theta_dk, (model.H_j[model._J_dk] >= model._z_inv_dk).astype(float))
    model.upstream_hydraulic_geometry()
    model.downstream_hydraulic_geometry()
    records = []
    for side in ['u','d']:
        suffix = side+'k'
        ilinks = model._i_1k if side=='u' else model._i_nk
        inodes = model._I_1k if side=='u' else model._I_Np1k
        supernodes = getattr(model,'_J_'+suffix)
        zinvs = getattr(model,'_z_inv_'+suffix)
        theta = getattr(model,'_theta_'+suffix)
        flows = getattr(model,'Q_'+suffix)
        coefficients = getattr(model,'_C_'+suffix)
        points,names,shapes,g,codes = endpoint_inputs(converted,model,side)
        assert np.allclose(zinvs,[p['z_inv'] for p in points],rtol=0,atol=1e-12)
        old = {k:getattr(model,'_'+k+'_'+suffix).copy() for k in GEOMETRY_METRICS}
        # 调同一Numba函数，仅用逐端面数组代替相邻内部单元数组；h、theta、H、z全不变。
        new = {k:np.zeros(model.NK) for k in GEOMETRY_METRICS}
        zeros = np.zeros(model.NK)
        numba_boundary_geometry(new['A'],new['Pe'],new['R'],new['B'],model.h_Ik,model.H_j,
            zinvs,theta,*g,zeros,zeros,zeros,zeros,codes,np.arange(model.NK),inodes,supernodes)
        # 并联闭口矩形的原有单孔湿周修复按同一孔数/单孔宽高原样应用。
        physical_scope['_correct_section'](model,new['A'],new['Pe'],new['R'],model._parallel_k,
            model._parallel_Nk,model._parallel_bk,model._parallel_Hk)
        gamma = gamma_uk if side=='u' else gamma_dk
        term_old = gamma(flows,coefficients,old['A'])*flows
        term_new = gamma(flows,coefficients,new['A'])*flows
        for k,p in enumerate(points):
            i=int(ilinks[k]);I=int(inodes[k]);J=int(supernodes[k])
            h_i=float(model.h_Ik[I]);h_j=float(theta[k]*(model.H_j[J]-zinvs[k]));mean=(h_i+h_j)/2
            internal_shape=str(converted['links'].loc[i,'shape'])
            midpoint=[float(getattr(model,f'_g{n}_ik')[i]) for n in (1,2,3)]
            endpoint=[float(v[k]) for v in g]
            closed=shapes[k]=='rect_closed'
            height_exceeded = not closed and max(h_i,h_j) > min(midpoint[0],endpoint[0])+1e-9
            dry = min(h_i,h_j)<-1e-9
            changed=internal_shape!=shapes[k] or any(abs(a-b)>1e-9 for a,b in zip(midpoint,endpoint))
            hydraulic_changed=any(abs(old[metric][k]-new[metric][k])>1e-9 for metric in GEOMETRY_METRICS)
            rec=dict(state=label,channel=names[k],superlink_id=k,side=side,
                physical_side=('source_node1' if (side=='u')==(converted['channel_discretization'][names[k]]['flow_sign']>0) else 'source_node2'),
                source_section=p['name'],source_interval=p['source_interval'],end_junction_id=I,superjunction_id=J,
                internal_link_id=i,adjacent_internal_link_length_m=float(model._dx_ik[i]),
                end_segment_length_m=float(getattr(model,'_dx_'+suffix)[k]),endpoint_bottom_m=float(zinvs[k]),
                boundary_depth_rule='(h_internal_endpoint + theta*(H_superjunction-z_endpoint))/2',
                internal_depth_m=h_i,superjunction_relative_depth_m=h_j,mean_depth_m=mean,
                theta=float(theta[k]),flow_m3s=float(flows[k]),end_head_C=float(coefficients[k]),
                old_shape=internal_shape,new_shape=shapes[k],old_g1_g2_g3=midpoint,endpoint_g1_g2_g3=endpoint,
                parameter_mismatch=changed,hydraulic_geometry_changed=hydraulic_changed,
                domain_status='closed_barrel_pressure_domain' if closed else ('outside_existing_contour' if height_exceeded or dry else 'inside_both_existing_contours'),
                endpoint_depth_above_height_m=0. if closed else max(h_i,h_j)-endpoint[0],
                used_mean_above_midpoint_height_m=0. if closed else mean-midpoint[0],
                used_mean_above_endpoint_height_m=0. if closed else mean-endpoint[0],
                old_geometry={metric:float(old[metric][k]) for metric in GEOMETRY_METRICS},
                endpoint_geometry={metric:float(new[metric][k]) for metric in GEOMETRY_METRICS},
                delta_geometry={metric:float(new[metric][k]-old[metric][k]) for metric in GEOMETRY_METRICS},
                old_end_head_term_signed_m=float(term_old[k]),endpoint_end_head_term_signed_m=float(term_new[k]),
                old_end_head_term_magnitude_m=float(abs(term_old[k])),endpoint_end_head_term_magnitude_m=float(abs(term_new[k])),
                head_term_magnitude_change_m=float(abs(term_new[k])-abs(term_old[k])),
                head_term_absolute_difference_m=float(abs(term_new[k]-term_old[k])))
            records.append(rec)
    assert before=={key:ARRSHA(getattr(model,key)) for key in before}
    return records


def summarize(rows):
    result={}
    for state in sorted({r['state'] for r in rows}):
        group=[r for r in rows if r['state']==state]
        opens=[r for r in group if r['new_shape']!='rect_closed']
        valid=[r for r in opens if r['domain_status']=='inside_both_existing_contours']
        closed=[r for r in group if r['new_shape']=='rect_closed']
        result[state]=dict(open_ends=len(opens),closed_ends=len(closed),
            open_parameter_mismatch=sum(r['parameter_mismatch'] for r in opens),
            open_hydraulic_geometry_changed=sum(r['hydraulic_geometry_changed'] for r in opens),
            open_inside_both_contours=len(valid),open_outside_contour=len(opens)-len(valid),
            inside_contour_geometry_changed=sum(r['hydraulic_geometry_changed'] for r in valid),
            inside_contour_max_abs_head_term_difference_m=max(r['head_term_absolute_difference_m'] for r in valid),
            closed_parameter_mismatch=sum(r['parameter_mismatch'] for r in closed),
            closed_hydraulic_geometry_changed=sum(r['hydraulic_geometry_changed'] for r in closed),
            top_inside_contour=sorted(valid,key=lambda r:r['head_term_absolute_difference_m'],reverse=True)[:12],
            western_zhao=[r for r in group if r['channel'] in ['POOL4-1-JJZ4DHX1','POOL4-2-DHX1JJZ5','siphon_DHX1-西赵河渠倒虹吸-pool4']])
    return result


source=json.loads((FROZEN/'共享输入/C07.json').read_text())['中线全线']
physical_scope=dict(np=np)
tree=ast.parse((DEPS/'并联结构物理修复.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_correct_section')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(DEPS/'并联结构物理修复.py'),'exec'),physical_scope)


def main():
    paths={'固定初态0s':FROZEN/'共享输入/初始化后状态_0s.npz',
           '原参数无同化30天末态':ROOT/'38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数/末态.npz'}
    watched=[Path(__file__),Path(helper.__file__),FROZEN/'共享输入/C07.json',DEPS/'并联结构物理修复.py',
        DEPS/'wxq_pipedream/wxq_to_superlink.py',DEPS/'wxq_pipedream/section_geometry.py',DEPS/'pipedream_solver/superlink.py',
        DEPS/'pipedream_solver/nsuperlink.py',DEPS/'pipedream_solver/ngeometry.py',*paths.values()]
    hashes={str(p):SHA(p) for p in watched}
    engine,converted=helper.physical_build(source)
    records=[]
    for label,path in paths.items():
        records.extend(compare(engine.model,converted,label,path))
    summary=summarize(records)
    minimal=dict(description='零长度端面，矩形实际端断面宽27.6m，邻接过渡单元中点b=24.8m/m=1.625；两端深度都7m，断面高均9.02m，Q=195m3/s、C=0.20151121全不变',
        h_internal_m=7.,h_superjunction_relative_m=7.,height_m=9.02,endpoint_rect_width_m=27.6,
        midpoint_bottom_width_m=24.8,midpoint_slope=1.625,endpoint_area_m2=27.6*7.,
        midpoint_area_m2=(24.8+1.625*7.)*7.)
    minimal['endpoint_head_term_m']=0.20151121*195**2/(2*9.81*minimal['endpoint_area_m2']**2)
    minimal['midpoint_head_term_m']=0.20151121*195**2/(2*9.81*minimal['midpoint_area_m2']**2)
    output=dict(scope='0步代数核查；126开渠252端面及105闭口结构210端面；只比较两个已保存无同化状态，不代表连续30天所有时刻',
        hydraulic_steps=0,source_hashes=hashes,source_hashes_unchanged=all(SHA(p)==v for p,v in hashes.items()),
        summary=summary,minimal_reproduction=minimal,
        findings={'classification':'当前WXQ变断面适配丢失端面独立几何；Pipedream分段常断面默认近似沿用至零长度端面。标准与Numba后端一致不能排除该一致性缺口。',
            'call_chain':['parse_wxq_json生成真实端点metadata和内部连接中点几何','HydraulicEngine把links中点g1/g2/g3传给NumbaSuperLink','upstream/downstream_hydraulic_geometry把相邻内部link索引传给numba_boundary_geometry','边界函数使用真实端点z_inv和同一平均水深，但shape/g1/g2/g3取内部中点','A_uk/A_dk进入gamma_uk/gamma_dk的C*Q|Q|/(2gA²)，B仅在端段长度非零时参与对应端段储水；本模型端段长度均0'],
            'unchanged':['n','C','h/H/Q','theta','并联孔数/单孔宽高','内部连接几何','半段/端段长度'],
            'domain_rule':'开渠两侧深度均在旧中点轮廓及端面轮廓内才列为域内；域外仅保留数值诊断，不宣称真实水头改进。闭口管单列压力水头，不按开敞高度拒绝。',
            'hydraulic_effect_scope':'水头项绝对差是在同一保存状态、同一Q和C下的代数变化，不是演进后水位变化或全月MAE收益。'},
        minimal_candidate={'scope':'41独立适配钩子或转换输出增加端部专用shape/g1/g2/g3，内部中点数组保持原样',
            'mapping':'按channel_discretization真实solver顺序取首/末point；不能按原node1/node2忽略flow_sign',
            'implementation':'专用端面数组以0..NK-1索引调用同一numba_boundary_geometry；保留原平均深度和theta。只覆盖开渠端部A/Pe/R/B；闭口105保持原物理孔数修复。',
            'avoid':'不能临时覆盖_g1_ik等内部数组，也不延长端段、改C或用另一个水深规则混入本候选。',
            'validation_before_running':['同断面渠道及105闭口结构应无差异','变断面端点应与源endpoint代数一致','所有内部A/B/Pe/R、h/H/Q、n/C逐值不变','原基线端部域外现象仍要独立报告，不能因切换几何绕过g1上界','需主进程确认后才写候选水力运行，不立即全月']},records=records)
    assert output['source_hashes_unchanged']
    write('端部几何来源核对.json',output)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} | {
        'old_A_m2':r['old_geometry']['A'],'endpoint_A_m2':r['endpoint_geometry']['A'],
        'old_B_m':r['old_geometry']['B'],'endpoint_B_m':r['endpoint_geometry']['B'],
        'old_Pe_m':r['old_geometry']['Pe'],'endpoint_Pe_m':r['endpoint_geometry']['Pe']} for r in records]).to_csv(HERE/'端部逐状态几何差异.csv',index=False)
    print(json.dumps({state:{k:v for k,v in row.items() if k not in ['top_inside_contour','western_zhao']} for state,row in summary.items()},ensure_ascii=False))


if __name__=='__main__':
    main()
