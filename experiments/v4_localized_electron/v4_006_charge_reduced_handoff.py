"""Eliminate the native cyclic phase increments in the finite-hop residual.

For arbitrary shape endpoints a,b and their two common U(1) phases phi,
the interval phase charge is d_phi Ld(a,R_phi b). Solve this pair of native
momentum equations for each preceding/following interval. The remaining
DEL residual is the shape equation after eliminating phases; no phase
reset, Dirac phase, target force, or new action is introduced. The sampled
shape paths are diagnostics only, not a free trajectory or a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_current_action_relative_orbit import shift
from v4_006_phase_corrected_handoff import evaluate_fourier, unpack


def rotation(q, phases):
    a=q.reshape(5,5,5,4,3);cp,cc=phases.cos();sp,sc=phases.sin()
    return torch.stack((cp*a[...,0,:]-sp*a[...,1,:],sp*a[...,0,:]+cp*a[...,1,:],
                        cc*a[...,2,:]-sc*a[...,3,:],sc*a[...,2,:]+cc*a[...,3,:]),dim=-2).reshape(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--dt',type=float,default=1/256);p.add_argument('--period-updates',type=int,default=17564)
    p.add_argument('--axis',type=int,default=0);p.add_argument('--sign',type=int,default=-1);p.add_argument('--nodes',type=int,default=5)
    p.add_argument('--shape',choices=('local','normalized_local','fourier'),default='normalized_local');p.add_argument('--warm-start',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt;n=a.period_updates
    rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64);grid=ref.reshape(5,5,5,12)
    vac=grid[0,0,0].clone();vac[:6]=0;scale=torch.linalg.vector_norm(ref-vac.repeat(125))
    theta=float(np.arccos(1-(dt*rank.R)**2/2));initial_phase=torch.tensor([theta,0.],dtype=ref.dtype)
    def ell(left,right,phases):return action.interval(left,rotation(right,phases),dt)
    grad=torch.func.grad(ell,argnums=2)
    Q=grad(ref,ref,initial_phase).detach().numpy();qscale=max(abs(Q[0]),1e-300)
    times=torch.arange(a.nodes,dtype=ref.dtype)/a.nodes;phase_guess=initial_phase.clone()
    if a.warm_start:
        previous=np.load(a.warm_start);nodes=evaluate_fourier(torch.tensor(previous['nodes'],dtype=ref.dtype),times);n=int(round(float(previous['period_updates'])));phase_guess[0]=float(previous['phase_per_step'])
    elif a.shape=='fourier':nodes=ref.repeat(a.nodes,1)
    else:
        translated=torch.roll(grid,a.sign,a.axis).reshape(-1);mix=(1-torch.cos(torch.pi*times))/2
        shapes=(1-mix[:,None])*ref+mix[:,None]*translated
        if a.shape=='normalized_local':
            sh=shapes.reshape(a.nodes,125,12);target=torch.linalg.vector_norm(grid[...,:6]);sh[...,:6]*=(target/torch.linalg.vector_norm(sh[...,:6],dim=(1,2)))[:,None,None]
        nodes=shift(shapes,-a.sign*times,a.axis)
    def q_at(t):return shift(evaluate_fourier(nodes,t),a.sign*t,a.axis)
    def solve_phase(left,right):
        def fun(value):return (grad(left,right,torch.tensor(value,dtype=ref.dtype)).detach().numpy()-Q)/qscale
        candidate=root(fun,phase_guess.numpy(),method='hybr',options={'xtol':1e-10,'maxfev':40});defect=float(np.linalg.norm(fun(candidate.x)))
        return torch.tensor(candidate.x,dtype=ref.dtype),defect,int(candidate.nfev)
    def powers(delta):
        psi,chi=unpack(torch.as_tensor(delta));ps=float(torch.sum(abs(psi)**2));cs=float(torch.sum(abs(chi)**2))
        return {'acceleration_defect':float(np.linalg.norm(delta)/float(scale)),'Psi_power_fraction':ps/max(ps+cs,1e-300)}
    F=np.load(a.tangent_cache)['following'];factor=lu_factor(F)
    rows=[]
    for t in times:
        t=t[None];left,current,right=q_at(t-1/n)[0],q_at(t)[0],q_at(t+1/n)[0]
        base=action.del_residual(rotation(left,-phase_guess),current,rotation(right,phase_guess),dt).detach().numpy()
        pv,pdef,piter=solve_phase(left,current);fv,fdef,fiter=solve_phase(current,right)
        actual=action.del_residual(rotation(left,-pv),current,rotation(right,fv),dt).detach().numpy()
        d0=lu_solve(factor,base)/dt**2;dr=lu_solve(factor,actual)/dt**2
        q_in=grad(left,current,phase_guess).detach().numpy();q_out=grad(current,right,phase_guess).detach().numpy()
        row={'fraction':float(t[0]),'baseline':powers(d0),'charge_reduced':powers(dr),
             'baseline_incoming_charge_relative_defect':((q_in-Q)/qscale).tolist(),'baseline_outgoing_charge_relative_defect':((q_out-Q)/qscale).tolist(),
             'incoming_phase':pv.tolist(),'outgoing_phase':fv.tolist(),'incoming_phase_charge_defect':pdef,'outgoing_phase_charge_defect':fdef,
             'phase_solve_evaluations':[piter,fiter]}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'status':'V4_006 native phase-momentum elimination diagnostic','action_sha256':sha256(Path(action.__file__)),'tangent_cache_sha256':sha256(a.tangent_cache),
            'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),'dt':dt,'axis':a.axis,'sign':a.sign,'period_updates':n,'shape':a.shape,
            'phase_reference':phase_guess.tolist(),'native_rest_charge':Q.tolist(),'rows':rows,
            'baseline_maximum_acceleration_defect':max(r['baseline']['acceleration_defect'] for r in rows),
            'charge_reduced_maximum_acceleration_defect':max(r['charge_reduced']['acceleration_defect'] for r in rows),
            'claim_boundary':'Exact native charge roots at sampled shape intervals; the residual is still the unrestricted native DEL, neither a native update nor a finite-hop pass.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
