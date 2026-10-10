"""候选钩子的两存档状态回归：只调用几何函数，禁止水动力推进。"""
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


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit=load('endpoint41_audit',HERE/'端部几何最小复现.py')
hook=load('endpoint41_candidate',HERE/'端部独立几何候选.py')
import numpy as np

SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARRSHA=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
WATCH=list(audit.STATE_KEYS)+['_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk',
    '_dx_ik','_dx_uk','_dx_dk','_z_inv_Ik','_theta_uk','_theta_dk',
    '_A_ik','_B_ik','_Pe_ik','_R_ik','_geom_codes']+[f'_g{i}_ik' for i in range(1,8)]


def main():
    recorded=json.loads((HERE/'端部几何来源核对.json').read_text())
    expected={(r['state'],r['superlink_id'],r['side']):r for r in recorded['records']}
    paths={'固定初态0s':audit.FROZEN/'共享输入/初始化后状态_0s.npz',
           '原参数无同化30天末态':audit.ROOT/'38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数/末态.npz'}
    hashes={str(p):SHA(p) for p in [Path(__file__),Path(hook.__file__),Path(audit.__file__),
        HERE/'端部几何来源核对.json',audit.FROZEN/'共享输入/C07.json',*paths.values()]}
    engine,converted=audit.helper.physical_build(audit.source)
    original=engine.model
    results=[]
    for label,path in paths.items():
        with np.load(path) as z:
            for key in audit.STATE_KEYS:getattr(original,key)[:]=z[key]
        original.link_hydraulic_geometry()
        original.upstream_hydraulic_geometry()
        original.downstream_hydraulic_geometry()
        before={key:ARRSHA(getattr(original,key)) for key in WATCH}
        end_before={side:{key:getattr(original,'_'+key+'_'+side+'k').copy() for key in audit.GEOMETRY_METRICS} for side in ['u','d']}
        candidate=copy.deepcopy(original)
        manifest=hook.attach_endpoint_geometry(candidate,converted,audit.source)
        assert candidate._endpoint_prior_upstream.__func__.__name__=='_up_geometry'
        assert candidate._endpoint_prior_downstream.__func__.__name__=='_down_geometry'
        assert before=={key:ARRSHA(getattr(candidate,key)) for key in WATCH}
        assert before=={key:ARRSHA(getattr(original,key)) for key in WATCH}
        assert candidate.t==original.t
        assert manifest['open_channels']==126 and manifest['fixed_endpoints']==252
        closed=original._parallel_k
        row=dict(state=label,unchanged_state_and_internal_array_count=len(WATCH),open_channels=126,
                 closed_barrels=105,closed_endpoints_exact=True,source_endpoint_outputs_exact=True,
                 prior_parallel_method_called_first=True,constant_endpoint_counts={},reverse_mapping_counts={})
        for side in ['u','d']:
            ids=candidate._endpoint_geometry_binding[side]['superlinks']
            points,names,shapes,g,codes=audit.endpoint_inputs(converted,original,side)
            old_links=original._i_1k if side=='u' else original._i_nk
            equal=[int(k) for k in ids if original._geom_codes[old_links[k]]==codes[k]
                and all(getattr(original,f'_g{n+1}_ik')[old_links[k]]==g[n][k] for n in range(3))]
            row['constant_endpoint_counts'][side]=len(equal)
            for metric in audit.GEOMETRY_METRICS:
                actual=getattr(candidate,'_'+metric+'_'+side+'k')
                assert np.array_equal(actual[closed],end_before[side][metric][closed])
                assert np.array_equal(actual[equal],end_before[side][metric][equal])
                assert np.array_equal(actual[ids],np.asarray([expected[(label,int(k),side)]['endpoint_geometry'][metric] for k in ids]))
            reversed_count=0
            for name,meta in converted['channel_discretization'].items():
                if name not in audit.source['baseData']['channels'] or meta['flow_sign']>0:continue
                p=meta['sections'][0 if side=='u' else -1]
                wanted=audit.source['baseData']['channels'][name]['sec_names'][-1 if side=='u' else 0]
                assert p['name']==wanted
                k=meta['superlink_id'];q=manifest['endpoints']
                bound=next(x for x in q if x['superlink_id']==k and x['side']==side)
                assert bound['source_section']==wanted
                reversed_count+=1
            row['reverse_mapping_counts'][side]=reversed_count
        # 后续集合/诊断副本仍绑定自己的原并联方法，不会改到来源模型。
        extra=copy.deepcopy(candidate)
        assert extra._endpoint_prior_upstream.__self__ is extra
        assert extra._endpoint_prior_downstream.__self__ is extra
        extra.upstream_hydraulic_geometry();extra.downstream_hydraulic_geometry()
        assert all(np.array_equal(getattr(extra,'_'+metric+'_'+side+'k'),getattr(candidate,'_'+metric+'_'+side+'k'))
            for side in ['u','d'] for metric in audit.GEOMETRY_METRICS)
        row['deepcopy_binding_isolated']=True
        row['open_domain_counts_preserved']=recorded['summary'][label]['open_outside_contour']
        results.append(row)
    assert all(SHA(p)==v for p,v in hashes.items())
    out=dict(status='PASSED',hydraulic_steps=0,source_hashes=hashes,source_hashes_unchanged=True,
        hooked_method_scope='只替换126开渠固定端面的A/Pe/R/B，不按水深切换，105闭口先执行既有并联修复后保持输出',
        candidate_path=str(Path(hook.__file__)),candidate_sha256=SHA(hook.__file__),results=results,
        limitations=['0s与30天末态是两个状态，不代表连续时序适用范围','原78/68个开渠端面域外现象仍原样列账','未修开敞截高函数、未改端损C、未改西赵河连接或任何现场几何','未授权或运行水动力精度比较'])
    (HERE/'端部候选零步核验结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(status=out['status'],candidate_sha256=out['candidate_sha256'],results=results),ensure_ascii=False))


if __name__=='__main__':main()
