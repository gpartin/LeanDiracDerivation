"""Rebuild or directly check the actual frozen V4 action's DEL blocks.

The historical cache archives are intentionally not required in git. The
small physical rest seed is tracked. Full derivation uses batched reverse
derivatives; --existing verifies deterministic independent JVPs instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as spectrum
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True);p.add_argument('--rest-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--existing',type=Path);p.add_argument('--chunk',type=int,default=8)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt
    config={'B':native_time.B,'link':action.LINK_SOURCE,'scale':action.SCALE,'spacetime':action.SPACETIME_PLAQUETTE_SCALE,
            'parallel_matched':action.PARALLEL_MATCHED_SCALE,'parallel_potential':action.PARALLEL_MATCHED_POTENTIAL_SCALE,
            'optional':[action.PLAQUETTE_SCALE,action.SYMPLECTIC_PLAQUETTE_SCALE,action.PARALLEL_OVERLAP_SCALE,
                        action.PARALLEL_FACE_SCALE,action.TRIANGLE_DELTA_SCALE,action.MATERIAL_AREA_SCALE], 'chi_payload':action.CHI_PAYLOAD}
    expected={'B':1197.,'link':'joint_face_laplacian_plus_frame_oscillator','scale':1197.,'spacetime':6/361,
              'parallel_matched':1.,'parallel_potential':1.,'optional':[0.]*6,'chi_payload':1.}
    if config!=expected:raise ValueError(f'action configuration differs: {config}')
    rest=np.load(a.rest_cache)['rest'];theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2));rotation=spectrum.rotation_matrix(rest.size,theta)
    layers=tuple(torch.tensor(x,dtype=torch.float64) for x in (rotation.T@rest,rest,rotation@rest)); names=('previous','current','following')
    blocks={}; rows=[];rng=np.random.default_rng(4002002)
    probes=rng.normal(size=(rest.size,2));probes/=np.linalg.norm(probes,axis=0)
    cache=np.load(a.existing) if a.existing else None
    for i,name in enumerate(names):
        def function(x):
            trial=list(layers);trial[i]=x
            return action.del_residual(*trial,dt)
        block=(cache[name] if cache is not None else torch.func.jacrev(function,chunk_size=a.chunk)(layers[i]).detach().numpy())
        if not np.isfinite(block).all() or block.shape!=(rest.size,rest.size):raise ValueError(name)
        if cache is None:blocks[name]=block
        for vec in probes.T:
            _,value=torch.func.jvp(function,(layers[i],),(torch.tensor(vec,dtype=torch.float64),))
            value=value.detach().numpy();error=float(np.linalg.norm(block@vec-value)/np.linalg.norm(value));rows.append({'block':name,'relative_error':error})
        print(name,rows[-2:],flush=True)
    if cache is None:
        a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**blocks);cache_path=a.output;report_path=a.output.with_suffix('.check.json')
    else:cache_path=a.existing;report_path=a.output
    maximum=max(r['relative_error'] for r in rows)
    result={'status':'NATIVE_DEL_JVP_AGREEMENT' if maximum<1e-8 else 'CACHE_MISMATCH','dt':dt,'action_configuration':config,
            'action_sha256':sha256(Path(action.__file__)),'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),
            'tangent_cache_sha256':sha256(cache_path),'maximum_relative_error':maximum,'rows':rows,'rebuild':cache is None}
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if maximum>=1e-8:raise ValueError('DEL cache does not match frozen action')


if __name__=='__main__':main()
