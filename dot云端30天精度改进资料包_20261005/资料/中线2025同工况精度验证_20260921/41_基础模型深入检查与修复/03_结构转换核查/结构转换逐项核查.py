"""只构模和检查实际数组；不推进水力时间，不改冻结模型。"""
from pathlib import Path
import ast, copy, hashlib, json, os, sys, types

sys.dont_write_bytecode = True
os.environ['NUMBA_CACHE_DIR'] = '/private/tmp/pipedream_frozen_41_numba'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / '26_冻结基线与后续归因'
DEPS = FROZEN / '共享依赖'
sys.path.insert(0, str(DEPS))
import numpy as np
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
UP = 'POOL4-1-JJZ4DHX1'
DOWN = 'POOL4-2-DHX1JJZ5'
SIPHON = 'DHX1-西赵河渠倒虹吸-pool4'


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def candidate(source):
    """恢复两端归属。新进口高度9.02m仅保留冻结轮廓假设，不冒充实测。"""
    result = copy.deepcopy(source)
    bd = result['baseData']
    assert 'q4s15' not in bd['sections']
    assert bd['channels'][UP]['sec_names'][-1] == 'q4s16'
    assert bd['channels'][DOWN]['sec_names'][0] == 'q4s17'
    inlet = copy.deepcopy(bd['sections']['q4s16'])
    delta = float(bd['siphons'][SIPHON]['inlet_zb']) - 135.45
    inlet.update(name='q4s15', location=69605.0)
    inlet['yz'] = [[x, z + delta] for x, z in inlet['yz']]
    bd['sections']['q4s15'] = inlet
    bd['channels'][UP]['sec_names'][-1] = 'q4s15'
    bd['channels'][DOWN]['sec_names'].insert(0, 'q4s16')
    return result


def physical_build(source):
    norms = json.loads((FROZEN / '共享输入/2025开度归一化上限.json').read_text())
    coeffs = json.loads((FROZEN / '共享输入/Pipedream闸流系数.json').read_text())
    c = parse_wxq_json(source, gate_max_openings=norms, section_geometry_mode='safe', minimum_initial_depth_m=7.)
    c['superjunctions']['h_0'] = 7.
    c['superjunctions']['max_depth'] = np.maximum(c['superjunctions'].max_depth.to_numpy(), 7.)
    c['orifices']['C'] = c['orifices'].name.map(coeffs).fillna(c['orifices'].C)
    for row in c['siphons_info']:
        k = row['superlink_id']
        mask = c['links'].k == k
        old = c['links'].loc[mask, 'dx'].sum()
        factor = row['physical_length'] / old
        c['links'].loc[mask, 'dx'] *= factor
        c['superlinks'].loc[c['superlinks'].id == k, 'dx'] = row['physical_length']
        for point in c['channel_discretization']['siphon_' + row['name']]['sections']:
            point['distance'] *= factor
    engine = PipedreamHydraulicEngine(c, dt=15., backend='numba')
    engine.model._method = 'lsq'
    path = DEPS / '并联结构物理修复.py'
    names = {'_correct_section', '_link_geometry', '_up_geometry', '_down_geometry', '_link_coeffs', 'attach_parallel_physics'}
    selected = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, types=types)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    scope['attach_parallel_physics'](engine, c)
    return engine, c


def summarize(source, c, model):
    bd = source['baseData']
    length_checks = []
    for name, channel in bd['channels'].items():
        locations = np.asarray([bd['sections'][k]['location'] for k in channel['sec_names']], float)
        diff = np.diff(locations)
        k = c['channel_discretization'][name]['superlink_id']
        length_checks.append(dict(channel=name, source_length_m=float(np.abs(diff).sum()),
            array_length_m=float(model._dx_ik[model._ki == k].sum()),
            monotone=bool(np.all(diff > 0) or np.all(diff < 0))))
    directional = []
    for name, meta in c['channel_discretization'].items():
        k = meta['superlink_id']
        link_ids = np.flatnonzero(model._ki == k)
        ids = [p['junction_id'] for p in meta['sections']]
        expected_z = [p['z_inv'] for p in meta['sections']]
        directional.append(dict(name=name, flow_sign=meta['flow_sign'],
            max_bottom_error_m=float(np.max(np.abs(model._z_inv_Ik[ids] - expected_z))),
            contiguous_connections=bool(np.array_equal(model._Ik[link_ids], ids[:-1]) and np.array_equal(model._Ip1k[link_ids], ids[1:]))))
    local = []
    for name in [UP, 'siphon_' + SIPHON, DOWN]:
        meta = c['channel_discretization'][name]
        k = meta['superlink_id']
        row = c['superlinks'].loc[k]
        points = meta['sections']
        edge_points = points[-2:] if name == UP else points[:2] if name == DOWN else points
        edge_links = meta['internal_link_ids'][-1:] if name == UP else meta['internal_link_ids'][:1]
        local.append(dict(name=name, superlink_id=k, length_m=float(model._dx_ik[model._ki == k].sum()),
            inlet_offset_m=float(row.in_offset), outlet_offset_m=float(row.out_offset),
            internal_links=[dict(id=int(i), length_m=float(model._dx_ik[i]), shape=str(c['links'].loc[i, 'shape']),
                g1=float(model._g1_ik[i]), g2=float(model._g2_ik[i]), g3=float(model._g3_ik[i]),
                j0=int(model._Ik[i]), j1=int(model._Ip1k[i])) for i in edge_links],
            sections=[dict(name=p['name'], junction_id=p['junction_id'], source_interval=p['source_interval'],
                distance_m=p['distance'], bottom_m=p['z_inv'], height_m=p['height'], width_m=p['bottom_width'], slope=p['side_slope']) for p in edge_points]))
    return dict(channels=len(length_checks), sections=len(bd['sections']), links=int(model.Nik),
        source_array_length_max_error_m=max(abs(x['source_length_m']-x['array_length_m']) for x in length_checks),
        nonmonotone_channels=[x for x in length_checks if not x['monotone']],
        geometry_issues=c['section_geometry_issues'], geometry_fallbacks=len(c['section_geometry_fallbacks']),
        shape_counts=c['links']['shape'].value_counts().to_dict(),
        geometry_direction_failures=[x for x in directional if x['max_bottom_error_m']>1e-10 or not x['contiguous_connections']],
        reversed_superlinks=sum(x['flow_sign']<0 for x in directional),
        western_zhao=local)


def main():
    path = FROZEN / '共享输入/C07.json'
    before_sha = sha(path)
    source = json.loads(path.read_text())['中线全线']
    modified = candidate(source)
    patches = [
        dict(op='add', path='/baseData/sections/q4s15', value=modified['baseData']['sections']['q4s15'],
            source_cells=['渐变段!G21:J21', '倒虹吸!C2:F2'], conditional_fields=['yz顶部143.57m沿用原轮廓高度9.02m，未取得竣工顶部证据', 'x1/y1/x2/y2沿用已有端断面坐标，未补测地理坐标']),
        dict(op='replace', path='/baseData/channels/'+UP+'/sec_names/'+str(len(source['baseData']['channels'][UP]['sec_names'])-1), old='q4s16', value='q4s15'),
        dict(op='add', path='/baseData/channels/'+DOWN+'/sec_names/0', value='q4s16', source_cells=['渐变段!C22:F22', '倒虹吸!G2:J2'])]
    save('西赵河最小候选字段差分.json', dict(base_sha256=before_sha, patches=patches,
        unchanged=['原q4s16完整对象','nodes','siphons','gates','boundaries','timeData','糙率及闸流系数'],
        candidate_level='同一源表可闭合连接与底宽底高程；入口顶部仍为沿用假设，非工程定稿'))
    engine, converted = physical_build(source)
    baseline = summarize(source, converted, engine.model)
    engine2, converted2 = physical_build(modified)
    after = summarize(modified, converted2, engine2.model)
    parallel = []
    for row in converted['siphons_info']:
        ids = np.flatnonzero(engine.model._ki == row['superlink_id'])
        count, width, height = row['n_orifice'], row['width'], row['height']
        parallel.append(dict(name=row['name'], barrels=count,
            total_width_error_m=float(np.max(np.abs(engine.model._g2_ik[ids]-count*width))),
            length_error_m=float(engine.model._dx_ik[ids].sum()-row['physical_length']),
            distributed_K_error=float(engine.model._parallel_Ki[np.isin(engine.model._parallel_i,ids)].sum()-row['loss_coeff'])))
    modules = [{ 'module':name, 'path':str(Path(mod.__file__).resolve()), 'sha256':sha(mod.__file__)}
        for name,mod in sys.modules.items() if name.startswith(('wxq_pipedream','pipedream_solver')) and getattr(mod,'__file__',None)]
    assert all(Path(row['path']).is_relative_to(DEPS.resolve()) for row in modules)
    assert sha(path)==before_sha
    assert baseline['geometry_direction_failures']==after['geometry_direction_failures']==[]
    assert baseline['nonmonotone_channels']==after['nonmonotone_channels']==[]
    assert all(abs(x['total_width_error_m'])<1e-10 and abs(x['length_error_m'])<1e-8 and abs(x['distributed_K_error'])<1e-10 for x in parallel)
    save('转换与实际数组核验.json', dict(baseline=baseline,candidate=after,parallel_105_checks=parallel,
        runtime_modules=modules,source_sha256=before_sha,source_unchanged=True,hydraulic_steps=0,
        assumption_upper_level_m=143.57,assumption_requirement='新入口与插值段需逐时间步核对水位严格小于原轮廓上界；一旦越界，本候选不能用于工程精度判断。'))
    print(json.dumps(dict(baseline=baseline,candidate=after,hydraulic_steps=0),ensure_ascii=False))


if __name__ == '__main__':
    main()
