"""Exact rotating-frame charge and clock audit for the frozen V4 action.

For a U(1)-invariant discrete action define l(q,theta)=Ld(q,R_theta q).
The exact phase charge is d_theta l and the fixed-charge stationary
functional is -(l-theta*Q)/dt (a discrete Routh reduction, not a claim of
exact finite-step Hamiltonian conservation). Matched clock terms can vanish
at theta0 while contributing nonzero phase charge. This audit measures that
effect along two candidate receiver-preparation seeds; these are not
prescribed dynamics or a requirement to translate between sites rigidly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_current_action_relative_orbit import psi_clock
from v4_006_phase_corrected_handoff import charge


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--dt',type=float,default=1/256)
    p.add_argument('--axis',type=int,default=0);p.add_argument('--sign',type=int,default=-1);p.add_argument('--points',type=int,default=9);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt
    rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64);grid=ref.reshape(5,5,5,4,3);shifted=torch.roll(grid,a.sign,a.axis)
    theta0=float(np.arccos(1-(dt*rank.R)**2/2));theta=torch.tensor(theta0,dtype=ref.dtype)
    weights={'base_and_common_material':lambda l,r:action.incumbent.interval(l,r,dt),
             'neighbor_prong_velocity':lambda l,r:dt*action.SCALE*action.one_form(l,r,dt),
             'six_face_oscillator':lambda l,r:action.SCALE*action.face_frame_oscillator_interval(l,r,dt),
             'spacetime_face':lambda l,r:action.SPACETIME_PLAQUETTE_SCALE*action.spacetime_factorized_face_interval(l,r,dt),
             'parallel_clock_completion':lambda l,r:action.PARALLEL_MATCHED_SCALE*action.parallel_overlap_matched_interval(l,r,dt)}

    def ell(q,t):return action.interval(q,psi_clock(q[None,:],t)[0],dt)
    deriv=torch.func.grad(ell,argnums=1)
    def calc(q,v):
        t=torch.tensor(v,dtype=ref.dtype);return float(deriv(q,t).detach())
    qref=calc(ref,theta0);ref_l=float(ell(ref,theta));reference=-(ref_l-theta0*qref)/dt
    direct=charge(ref,psi_clock(ref[None,:],theta)[0],dt)[0]
    pnorm=torch.linalg.vector_norm(grid[...,:2,:]);rows=[]
    for variant in ('raw_local_mix','Psi_norm_preserved'):
        for f in np.linspace(0,1,a.points):
            q=((1-f)*grid+f*shifted).clone()
            if variant=='Psi_norm_preserved':q[...,:2,:]*=pnorm/torch.linalg.vector_norm(q[...,:2,:])
            q=q.reshape(-1);at0=calc(q,theta0)
            low,high=.125*theta0,4*theta0
            fl=calc(q,low)-qref;fh=calc(q,high)-qref
            root=brentq(lambda z:calc(q,z)-qref,low,high,xtol=1e-13) if fl*fh<=0 else None
            terms={}
            for name,fn in weights.items():
                def reduced(t):return fn(q,psi_clock(q[None,:],t)[0])
                terms[name]={'interval_value_at_rest_clock':float(reduced(theta).detach()),'phase_charge_at_rest_clock':float(torch.func.grad(reduced)(theta).detach())}
            row={'variant':variant,'fraction':float(f),'phase_charge_at_rest_clock':at0,'phase_charge_relative_to_rest':at0/qref,
                 'charge_sum_relative_error':abs(sum(x['phase_charge_at_rest_clock'] for x in terms.values())-at0)/max(abs(at0),1e-300),
                 'native_clock_terms':terms,'fixed_charge_phase_per_step':root,'fixed_charge_phase_ratio':None if root is None else root/theta0,
                 'fixed_charge_routhian_delta':None if root is None else -(float(ell(q,torch.tensor(root,dtype=ref.dtype)))-root*qref)/dt-reference,
                 'fixed_charge_relative_defect':None if root is None else abs(calc(q,root)-qref)/abs(qref)}
            rows.append(row);print(variant,round(float(f),3),'Q/Q0',row['phase_charge_relative_to_rest'],'theta/theta0',row['fixed_charge_phase_ratio'],flush=True)
    result={'status':'V4_006_NATIVE_FIXED_CHARGE_CLOCK_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),
            'dt':dt,'axis':a.axis,'sign':a.sign,'theta_rest':theta0,'phase_charge_rest':qref,'direct_Noether_charge_check_relative_error':abs(float(direct)-qref)/abs(qref),
            'fixed_charge_routhian_rest':reference,'rows':rows,'claim_boundary':'Exact action restriction and phase derivative along two diagnostic seed paths only. Neither path is assumed to be an orbit, and a Routhian extremum along these paths does not establish a universal pinning barrier.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
