"""41隔离候选：只恢复开渠端面几何来源，保留Numba原水深组合与既有并联修复。"""
from pathlib import Path
import hashlib
import types
import numpy as np
from pipedream_solver.nsuperlink import numba_boundary_geometry


def _replace_open_ends(model, side):
    binding = model._endpoint_geometry_binding[side]
    ids = binding['superlinks']
    suffix = side+'k'
    outputs = binding['outputs']
    # 端面几何固定绑定；只有原有水深与theta随模型演进，不按水深切换几何。
    numba_boundary_geometry(outputs['A'],outputs['Pe'],outputs['R'],outputs['B'],
        model.h_Ik,model.H_j,getattr(model,'_z_inv_'+suffix)[ids],getattr(model,'_theta_'+suffix)[ids],
        *binding['g'],binding['codes'],binding['local_indices'],binding['internal_nodes'],
        getattr(model,'_J_'+suffix)[ids])
    for key,values in outputs.items():
        getattr(model,'_'+key+'_'+suffix)[ids] = values


def _upstream(self, *args, **kwargs):
    # 先完整执行原端部方法（包括并联单孔湿周修复），只覆盖固定开渠集合。
    self._endpoint_prior_upstream(*args, **kwargs)
    _replace_open_ends(self, 'u')


def _downstream(self, *args, **kwargs):
    self._endpoint_prior_downstream(*args, **kwargs)
    _replace_open_ends(self, 'd')


def attach_endpoint_geometry(model, converted, source):
    """在已装配原并联物理修复的Numba模型上固定绑定；不改内部连接或水位状态。"""
    if hasattr(model, '_endpoint_geometry_binding'):
        raise ValueError('端部独立几何候选不得重复绑定')
    if not hasattr(model, '_parallel_native_class'):
        raise ValueError('须先装配既有并联结构物理修复')
    if model._parallel_native_class.__module__ != 'pipedream_solver.nsuperlink':
        raise ValueError('此隔离候选只覆盖已核验的Numba后端')
    code = {'rect_open':3,'trapezoidal':5}
    metas = [(name, converted['channel_discretization'][name]) for name in source['baseData']['channels']]
    metas.sort(key=lambda item:item[1]['superlink_id'])
    ids = np.asarray([meta['superlink_id'] for _,meta in metas],int)
    if np.intersect1d(ids,model._parallel_k).size:
        raise ValueError('开渠端面与闭口并联结构身份重叠')
    binding = {}
    audit = []
    for side in ['u','d']:
        points = [meta['sections'][0 if side=='u' else -1] for _,meta in metas]
        if any(point['shape_kind'] not in code for point in points):
            raise ValueError('当前候选只接受已核实矩形/梯形开敞端面，不近似不规则端面')
        nodes = np.asarray([p['junction_id'] for p in points],int)
        runtime_nodes = (model._I_1k if side=='u' else model._I_Np1k)[ids]
        if not np.array_equal(nodes,runtime_nodes):
            raise ValueError('端点metadata与实际solver端序不一致')
        if not np.allclose([p['z_inv'] for p in points],getattr(model,'_z_inv_'+side+'k')[ids],rtol=0,atol=1e-12):
            raise ValueError('端点metadata与实际端部床高不一致')
        g = [np.asarray([p[key] for p in points],float) for key in ['height','bottom_width','side_slope']]
        g += [np.zeros(len(ids)) for _ in range(4)]
        binding[side] = dict(superlinks=ids.copy(),internal_nodes=nodes,
            g=g,codes=np.asarray([code[p['shape_kind']] for p in points],int),
            local_indices=np.arange(len(ids)),outputs={k:np.zeros(len(ids)) for k in ['A','Pe','R','B']})
        audit.extend(dict(channel=name,superlink_id=meta['superlink_id'],side=side,
            source_section=p['name'],junction_id=p['junction_id'],shape=p['shape_kind'],
            g1=p['height'],g2=p['bottom_width'],g3=p['side_slope'],flow_sign=meta['flow_sign'])
            for (name,meta),p in zip(metas,points))
    model._endpoint_prior_upstream = model.upstream_hydraulic_geometry
    model._endpoint_prior_downstream = model.downstream_hydraulic_geometry
    model._endpoint_geometry_binding = binding
    model.upstream_hydraulic_geometry = types.MethodType(_upstream,model)
    model.downstream_hydraulic_geometry = types.MethodType(_downstream,model)
    model.upstream_hydraulic_geometry()
    model.downstream_hydraulic_geometry()
    model.endpoint_geometry_manifest = dict(revision='endpoint_source_geometry_41_v1',
        hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        open_channels=len(ids),fixed_endpoints=len(audit),closed_siphons_unchanged=True,
        depth_rule='原numba_boundary_geometry平均水深与theta不变',
        dynamic_geometry_switch=False,internal_midpoint_geometry_changed=False,
        original_parallel_geometry_called_first=True,endpoints=audit)
    return model.endpoint_geometry_manifest
