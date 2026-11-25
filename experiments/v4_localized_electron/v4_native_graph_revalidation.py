"""Recheck V4-002/004 using the same moving contrast graph as V4-003.

This is a native tangent revalidation, not a nonlinear stability theorem.
The previous layer is generated only by the selected native spectral graph,
and the full recurrence evolves freely for the declared duration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cubical_edge_shift_channels as cubical
import cube_incidence_extended_packet_gate as dirac
import canonical_noncoplanar_bifurcation_audit as native_time
import edge_snap_action_rank as rank
import relative_kinetic_probe as physical
from v4_003_moving_native_graph_audit import (
    HERE, get_action, spatial_lifts, spatial_shapes, orth, native_graph)
from v4_003_fixed_phase_space_readout import sha256
from v4_004_operational_spin_gate import spin_turn, complex_matrix_to_real


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--tangent-cache',type=Path,required=True)
    parser.add_argument('--rest-cache',type=Path,required=True)
    parser.add_argument('--cycles',type=float,default=100)
    parser.add_argument('--zero-secular-charge',action='store_true')
    parser.add_argument('--null-tolerance',type=float,default=1e-9)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();dt=args.dt
    rank.lock_inertia()
    rest=np.load(args.rest_cache)['rest'];advance=get_action(dt,args.tangent_cache,rest)
    lifts,_,bases=spatial_lifts(rest);raw=lifts['contrast'];shapes=spatial_shapes()
    source=np.column_stack([raw[k]@shapes for k in ('plus','minus')])
    q,_=orth(source); r=q.shape[1]
    if r!=source.shape[1]:raise ValueError('copy packet coordinate rank deficient')
    zero=np.zeros_like(q)
    g,diag=native_graph(q,advance(q,zero),advance(zero,q))
    if g is None:raise ValueError(f'native graph ambiguous: {diag}')
    coordinate=q.T@source
    spin=[]
    for axis in range(3):
        for angle in (np.pi/2,2*np.pi,4*np.pi):
            from scipy.linalg import block_diag
            copies=[np.kron(np.eye(shapes.shape[1]//8),complex_matrix_to_real(spin_turn(cubical,bases[k],axis,angle))) for k in ('plus','minus')]
            rot=block_diag(*copies)
            rotq=np.linalg.solve(coordinate.T,(coordinate@rot).T).T
            defect=float(np.linalg.norm(g@rotq-rotq@g)/np.linalg.norm(g@rotq))
            spin.append({'axis':axis,'angle':angle,'two_layer_graph_commutator':defect,
                         'commutator_per_dt':defect/dt})
    seeds=[]; cases=[]
    for copy in ('plus','minus'):
        alpha,beta=dirac.compressed_clifford(bases[copy])
        for axis in range(4):
            for sign in (-1,1):
                _,field,_,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                seeds.append(raw[copy]@dirac.real_coordinates(field))
                cases.append({'copy':copy,'axis':axis,'sign':sign})
    now=np.column_stack(seeds);before=q@g@(q.T@now)
    correction=None; charge_map=None; charge_initial=None; maximum_charge_drift=0.0
    if args.zero_secular_charge:
        # In co-rotating coordinates x_{n+1}=A x_n+B x_{n-1}.
        # Any left null vector l of K=I-A-B gives the conserved linear
        # quantity l^T(x_n+B x_{n-1}). Stationary right-null coordinates s
        # can carry only values l^T(I+B)s. Clear the incompatible part once
        # in the incoming layer (no evolution projection). The chosen
        # numerical null tolerance remains explicitly diagnostic.
        identity=np.eye(rest.size); B=advance(np.zeros_like(identity),identity)
        K=(identity-advance(identity,identity))/dt**2
        l,s,vh=np.linalg.svd(K,full_matrices=True)
        keep=s<args.null_tolerance; L=l[:,keep]; S=vh[keep,:].T
        stationary=L.T@(identity+B)@S
        u,sval,_=np.linalg.svd(stationary,full_matrices=True)
        null=sval<1e-8*max(float(sval[0]),dt)
        W=u[:,null]
        charge_map=W.T@L.T
        incoming=charge_map@B
        original=before.copy()
        invariant=charge_map@(now+B@before)
        before-=np.linalg.lstsq(incoming,invariant,rcond=1e-10)[0]
        charge_initial=charge_map@(now+B@before)
        correction={'approximate_stationary_null_dimension':int(np.sum(keep)),
          'incompatible_charge_rank':int(np.sum(null)),
          'null_tolerance':args.null_tolerance,'stationary_null_residual':float(np.linalg.norm(K@S)/max(np.sqrt(S.shape[1]),1)),
          'maximum_initial_incompatible_charge':float(np.max(np.linalg.norm(invariant,axis=0))),
          'maximum_corrected_incompatible_charge':float(np.max(np.linalg.norm(charge_initial,axis=0))),
          'maximum_initial_velocity_relative_change':float(np.max(np.linalg.norm(before-original,axis=0)/np.maximum(np.linalg.norm(now-original,axis=0),1e-300)))}
    initial=np.sqrt(np.sum(now**2+before**2,axis=0))
    velocity_initial=np.sqrt(np.sum(((now-before)/dt)**2,axis=0)+np.sum(now**2,axis=0))
    maximum=np.ones(len(cases));minimum=np.ones(len(cases));maximum_velocity=np.ones(len(cases))
    embedding=np.vstack((q,q@g));left=np.linalg.pinv(embedding,rcond=1e-12)
    steps=int(np.ceil(args.cycles*2*np.pi/native_time.CHI0/dt));samples=[]
    sampling=set(np.linspace(0,steps,11,dtype=int))
    penultimate=sorted(sampling)[-2]; late=None
    for step in range(steps+1):
        ratio=np.sqrt(np.sum(now**2+before**2,axis=0))/initial
        vratio=np.sqrt(np.sum(((now-before)/dt)**2,axis=0)+np.sum(now**2,axis=0))/velocity_initial
        maximum=np.maximum(maximum,ratio);minimum=np.minimum(minimum,ratio);maximum_velocity=np.maximum(maximum_velocity,vratio)
        if step==penultimate:late=now.copy()
        if step in sampling:
            state=np.vstack((now,before));residual=state-embedding@(left@state)
            samples.append({'step':step,'time':step*dt,'maximum_norm_ratio':float(np.max(ratio)),
                            'maximum_velocity_aware_ratio':float(np.max(vratio)),
                            'maximum_readout_complement':float(np.max(np.linalg.norm(residual,axis=0)/np.linalg.norm(state,axis=0)))})
            if charge_map is not None:
                maximum_charge_drift=max(maximum_charge_drift,float(np.max(np.linalg.norm(charge_map@(now+B@before)-charge_initial,axis=0))))
        if step==steps:break
        following=advance(now,before);before,now=now,following
    for index,case in enumerate(cases):
        case.update(maximum_phase_space_norm_ratio=float(maximum[index]),minimum_phase_space_norm_ratio=float(minimum[index]),maximum_velocity_aware_norm_ratio=float(maximum_velocity[index]))
    worst=int(np.argmax(maximum))
    slope=(now[:,worst]-late[:,worst])/((steps-penultimate)*dt)
    p,c=physical.unpack_raw(slope,5)
    refp,refc=physical.unpack_raw(rest,5)
    exterior=np.linalg.norm(refp,axis=-1)<1e-12
    power=float(np.linalg.norm(p)**2+np.linalg.norm(c)**2)
    radial=np.real(np.sum(refc.conj()*c,axis=-1))/np.maximum(np.linalg.norm(refc,axis=-1),1e-300)
    phasep=rank.pack(1j*refp,np.zeros_like(refc))
    phasec=rank.pack(np.zeros_like(refp),1j*refc)
    sym,_=orth(np.column_stack((phasep,phasec)))
    slope_result={'worst_case':cases[worst], 'Psi_power_fraction':float(np.linalg.norm(p)**2/power),
       'chi_power_fraction':float(np.linalg.norm(c)**2/power), 'exterior_chi_power_fraction':float(np.linalg.norm(c[exterior])**2/power),
       'chi_radial_power_fraction':float(np.linalg.norm(radial)**2/power),
       'global_phase_power_fraction':float(np.linalg.norm(sym.T@slope)**2/power),
       'stationary_slope_native_update_defect_per_dt_squared':float(np.linalg.norm(advance(slope,slope)-slope)/np.linalg.norm(slope)/dt**2)}
    if correction is not None:correction['maximum_incompatible_charge_drift']=maximum_charge_drift
    result={'status':'NATIVE_MOVING_GRAPH_REVALIDATION','dt':dt,'cycles':args.cycles,'steps':steps,'action_sha256':sha256(HERE/'live_clifford_bond_action.py'),'tangent_cache_sha256':sha256(args.tangent_cache),'rest_cache_sha256':sha256(args.rest_cache),'native_branch_diagnostics':diag,'one_time_native_charge_preparation':correction,'spin':spin,'cases':cases,'samples':samples,'late_secular_slope':slope_result,'maximum_norm_ratio':float(np.max(maximum)),'minimum_norm_ratio':float(np.min(minimum)),'maximum_velocity_aware_norm_ratio':float(np.max(maximum_velocity)), 'claim_boundary':'Full unrestricted tangent only, using the same native moving graph or explicitly labeled one-time conserved-charge diagnostic. The underlying exact cubical spin identity remains independent; graph commutation is tested rather than assumed.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','spin','samples')},indent=2))


if __name__=='__main__':main()
