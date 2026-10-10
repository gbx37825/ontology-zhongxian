"""不拟合实测的性质核验：静水保持及改变绝对高程零点后的同物理过程。"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import types
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
B = R/'26_冻结基线与后续归因'
DEPS = B/'共享依赖'
R41 = R/'41_基础模型深入检查与修复'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def source_model(datum, bed_reversed):
    # 人工定义，非任何现场断面；特意令主节点底不同于端断面底，覆盖非零offset。
    beds = [1., .5] if bed_reversed else [.5, 1.]
    source = dict(baseData=dict(nodes={
        'left':dict(zb=datum-.4,Amin=100.,nodeType=1),
        'right':dict(zb=datum+.4,Amin=100.,nodeType=1)},
        channels={'reach':dict(node1='left',node2='right',sec_names=['a','b'],nc=.025)},
        sections={},gates={'gate':dict(node1='left',node2='right',b=2.,c1=.65,zb=datum+.3)},
        siphons={},pumps={},turbines={},curves={}),
        initialData=dict(boundaries={'left':datum+3.,'right':datum+3.},channels={'local_Z':{}},
                         bleeders={},gates={},pumps={},turbines={},options={'timeStep':15}),
        timeControl=dict(startTime=0,endTime=450,calculateStep=15,outputStep=15),
        timeSeries=dict(seriesLink={},seriesData={}))
    for name, x, bed, width, slope, interp in [('a',0.,beds[0],6.,1.,3),('b',100.,beds[1],8.,0.,0)]:
        z = datum+bed; h=8.; run=slope*h
        source['baseData']['sections'][name]=dict(location=x,num_interp=interp,
             yz=[[0.,z+h],[run,z],[run+width,z],[2*run+width,z+h]])
    return source


def main():
    paths=[Path(__file__), R41/'01_求解器核查/面积储水与闸流最小复现.py',
           R41/'09_端部几何一致性核查/端部独立几何候选.py',
           R41/'10_端面几何锁参验证/运行端面几何配对.py',
           B/'共享输入/初始化后状态_0s.npz', R41/'10_端面几何锁参验证/endpoint/末态.npz']
    paths += list((DEPS/'pipedream_solver').glob('*.py'))+list((DEPS/'wxq_pipedream').glob('*.py'))
    watched={str(p):SHA(p) for p in paths}
    protocol=dict(scope='制造静水及坐标平移性质；不校准/评分实测，不作整月精度替代',
         static_datums_m=[0.,-200.,1000.],static_directions=[False,True],static_steps=30,
         dynamic_runtime_datum_shifts_m=[0.,-200.,1000.],dynamic_steps_per_case=1,
         static_head_tolerance_m=1e-7,static_flow_tolerance_m3s=1e-6,
         datum_head_tolerance_m=1e-7,datum_flow_tolerance_m3s=1e-5,
         dt_s=15.,num_iter=20,head_tol_m=.0015,flow_tol_m3s=1e-9,
         engineering_geometry_modified=False,source_sha256=watched)
    write('运行前性质约定.json',protocol)
    commonrun=load(R41/'10_端面几何锁参验证/运行端面几何配对.py','_42_properties_common')
    shared,common,advance,restore,full_engine,full_source,full_c=commonrun.build_runtime()
    hook=load(R41/'09_端部几何一致性核查/端部独立几何候选.py','_42_properties_endpoint')
    physical=shared.bind(DEPS/'并联结构物理修复.py',
         ['_correct_section','_link_geometry','_up_geometry','_down_geometry','_link_coeffs','attach_parallel_physics'],
         dict(np=np,types=types),'_42_properties_parallel')
    results=[]
    try:
        # 完全合成且都在轮廓内的静水，覆盖顺坡/逆坡、非零端部offset及矩形闸孔。
        for datum in protocol['static_datums_m']:
            for reverse in protocol['static_directions']:
                s=source_model(datum,reverse)
                for endfix in [False,True]:
                    c=shared.parse_wxq_json(s,section_geometry_mode='safe')
                    c['superjunctions'].loc[:,'h_0']=datum+3.-c['superjunctions'].z_inv
                    c['junctions'].loc[:,'h_0']=datum+3.-c['junctions'].z_inv
                    c['links'].loc[:,'Q_0']=0.;c['superlinks'].loc[:,'Q_0']=0.
                    e=shared.PipedreamHydraulicEngine(c,dt=15.,backend='numba');m=e.model;m._method='lsq'
                    physical.attach_parallel_physics(e,c)
                    if endfix:hook.attach_endpoint_geometry(m,c,s)
                    # 消除构造器微步的数值残留，只对合成解析初态赋值；不影响真实模型试验。
                    m.H_j[:]=datum+3.;m.h_Ik[:]=datum+3.-m._z_inv_Ik
                    for key in ['Q_ik','Q_uk','Q_dk','Q_o']:getattr(m,key)[:]=0.
                    m.t=0.;m.iter_count=0
                    mxH=mxQ=0.
                    for _ in range(30):
                        e.step(boundary_heads={'left':datum+3.,'right':datum+3.},gate_openings={'gate':.5},
                               dt=15.,num_iter=20,strict=True,flow_tol=1e-9)
                        mxH=max(mxH,float(np.max(abs(m.H_j-(datum+3.)))),float(np.max(abs(m.h_Ik+m._z_inv_Ik-(datum+3.)))))
                        mxQ=max(mxQ,*[float(np.max(abs(getattr(m,k)))) for k in ['Q_ik','Q_uk','Q_dk','Q_o']])
                    results.append(dict(kind='manufactured_static',datum_m=datum,reversed_bed=reverse,endpoint_hook=endfix,
                         steps=30,max_head_error_m=mxH,max_abs_flow_m3s=mxQ,
                         passed=mxH<=1e-7 and mxQ<=1e-6))
        # 真实保存状态只整体平移绝对坐标；水深、尺寸、坡度、offset、n/C和流量均不变。
        forcing=common.Forcing(full_engine,full_source,full_c)
        for name,path,t,endpoint in [('原始0s',B/'共享输入/初始化后状态_0s.npz',0.,False),
                ('端面候选30天末',R41/'10_端面几何锁参验证/endpoint/末态.npz',2592000.,True)]:
            with np.load(path) as z:states={k:z[k].copy() for k in commonrun.STATE_KEYS}
            base=copy.deepcopy(full_engine.model);restore(base,states);base.t=t;base.iter_count=0
            if endpoint:hook.attach_endpoint_geometry(base,full_c,full_source)
            values=[]
            for shift in protocol['dynamic_runtime_datum_shifts_m']:
                m=copy.deepcopy(base)
                for key in ['H_j','_z_inv_j','_z_inv_Ik','_z_inv_uk','_z_inv_dk']:getattr(m,key)[:]+=shift
                kw=forcing.at(t);kw['H_bc']=kw['H_bc'].copy();kw['H_bc'][np.isfinite(kw['H_bc'])]+=shift
                kw.update(dt=15.,num_iter=20,strict=True,head_tol=.0015,flow_tol=1e-9)
                m.step(**kw)
                out={k:getattr(m,k).copy() for k in commonrun.STATE_KEYS};out['H_j']-=shift
                values.append(out)
                diff={k:float(np.max(abs(out[k]-values[0][k]))) for k in commonrun.STATE_KEYS}
                results.append(dict(kind='saved_state_datum_shift',source_state=name,shift_m=shift,steps=1,
                    counterfactual_continuation_not_scored=True,max_differences=diff,
                    passed=max(diff['H_j'],diff['h_Ik'])<=1e-7 and max(diff[k] for k in ['Q_ik','Q_uk','Q_dk','Q_o'])<=1e-5))
        assert all(SHA(p)==h for p,h in watched.items())
        out=dict(status='PROPERTIES_CHECKED',all_passed=all(x['passed'] for x in results),cases=results,
                 source_sha256=watched,source_hashes_unchanged=True,base_model_accuracy_objective_complete=False,
                 scope='仅覆盖所列制造性质和两份真实保存状态单步坐标不变性；不证明结构资料或全月精度')
        write('性质核验结果.json',out)
        print(json.dumps(out,ensure_ascii=False),flush=True)
    finally:
        shutil.rmtree(shared.CACHE,ignore_errors=True)


if __name__=='__main__':main()
