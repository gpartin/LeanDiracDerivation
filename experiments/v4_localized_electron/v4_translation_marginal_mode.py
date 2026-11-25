"""Test whether native secular response could be the desired translation.

In rotating coordinates the linear update is x[n+1]=A*x[n]+B*x[n-1].
A drift x[n]=r+n*dt*v requires K*v=0 and K*r=-dt*(I+B)*v,
K=I-A-B. Test the physical full-field translation derivative, not a
spinor-only phase mode. Discrete finite hops can exist without this local
continuous drift; failure is not a finite-hop no-go.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import edge_snap_action_rank as rank
from v4_003_fixed_phase_space_readout import sha256
from v4_003_moving_native_graph_audit import (
    HERE,get_action,native_graph,orth,spatial_lifts,spatial_shapes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True)
    p.add_argument('--tangent-cache',type=Path,required=True);p.add_argument('--rest-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();dt=a.dt;rank.lock_inertia()
    rest=np.load(a.rest_cache)['rest'];adv=get_action(dt,a.tangent_cache,rest)
    lifts,translations,bases=spatial_lifts(rest);raw=lifts['contrast'];shape=spatial_shapes()
    q,_=orth(np.column_stack([raw[k]@shape for k in ('plus','minus')]))
    g,diag=native_graph(q,adv(q,np.zeros_like(q)),adv(np.zeros_like(q),q))
    if g is None:raise ValueError(diag)
    # Native spectral initialization, all signed-axis and diagonal probes.
    seeds=[];cases=[]
    for copy in ('plus','minus'):
        alpha,beta=dirac.compressed_clifford(bases[copy])
        for axis in range(4):
            for sign in (-1,1):
                _,field,_,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                seeds.append(raw[copy]@dirac.real_coordinates(field));cases.append((copy,axis,sign))
    now=np.column_stack(seeds);before=q@g@(q.T@now)
    eye=np.eye(rest.size);B=adv(np.zeros_like(eye),eye);K=(eye-adv(eye,eye))/dt**2
    u,s,vh=np.linalg.svd(K,full_matrices=True);rows=[]
    for tol in (1e-7,1e-9,1e-11):
        keep=s<tol;L=u[:,keep];S=vh[keep,:].T
        for axis in range(3):
            v=translations[:,axis];projected=S@(S.T@v);pv=np.linalg.norm(projected)
            rhs=(eye+B)@projected/dt
            rows.append({'tol':tol,'axis':axis,'null_dimension':int(np.sum(keep)),
                         'translation_null_overlap':float(pv),
                         'translation_stationary_defect':float(np.linalg.norm(K@v)),
                         'jordan_compatibility_defect':float(np.linalg.norm(L.T@rhs)/max(np.linalg.norm(rhs),1e-300))})
    total=int(np.ceil(100*2*np.pi/native_time.CHI0/dt));late=None
    for step in range(total+1):
        if step==int(.9*total):late=now.copy()
        if step==total:break
        after=adv(now,before);before,now=now,after
    slope=(now-late)/((total-int(.9*total))*dt);T,_=orth(translations)
    overlap=np.linalg.norm(T.T@slope,axis=0)/np.maximum(np.linalg.norm(slope,axis=0),1e-300)
    result={'status':'NATIVE_MARGINAL_TRANSLATION_DIAGNOSTIC','dt':dt,'action_sha256':sha256(HERE/'live_clifford_bond_action.py'),
            'tangent_cache_sha256':sha256(a.tangent_cache),'rest_cache_sha256':sha256(a.rest_cache),
            'nullspace_tests':rows,'secular_slope_translation_overlap':[{'copy':c,'axis':j,'sign':k,'overlap':float(v)} for (c,j,k),v in zip(cases,overlap)],
            'claim_boundary':'Local linear drift only, not an obstruction to finite integer-cell relative-periodic motion.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
