"""Finite relative-periodic collocation screen of the actual V4-002 action.

Use the existing exact translated endpoint, but no rigid intermediate path.
In a translating and clock-rotating chart the unknown shape is periodic;
its independent collocation layers are varied to satisfy the *unmodified*
native discrete action. The spatial Fourier shift is a coordinate chart for
the boundary condition, not a simulation rule. A sampled collocation root
would still require a full native free release and box/timestep refinement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import live_clifford_bond_action as action
import canonical_noncoplanar_bifurcation_audit as native_time
import edge_snap_action_rank as rank
from v4_003_fixed_phase_space_readout import sha256


HERE=Path(__file__).resolve().parent
SIDE=5


def shift(fields, amount, axis):
    grid=fields.reshape((-1,SIDE,SIDE,SIDE,12))
    dim=axis+1
    freq=torch.fft.fftfreq(SIDE,dtype=grid.dtype,device=grid.device)
    shape=[grid.shape[0],1,1,1,1];shape[dim]=SIDE
    angle=-2*torch.pi*amount.reshape(-1,1)*freq.reshape(1,-1)
    phase=torch.exp(1j*angle).reshape(shape)
    return torch.fft.ifft(torch.fft.fft(grid,dim=dim)*phase,dim=dim).real.reshape(fields.shape)


def psi_clock(fields,angle):
    u=fields.reshape(-1,SIDE,SIDE,SIDE,4,3)
    ar=angle.cos();ai=angle.sin()
    return torch.stack((ar*u[...,0,:]-ai*u[...,1,:],ai*u[...,0,:]+ar*u[...,1,:],u[...,2,:],u[...,3,:]),dim=-2).reshape(fields.shape)


def time_shift(values, fraction):
    freq=torch.fft.fftfreq(values.shape[0],d=1/values.shape[0],dtype=values.dtype,device=values.device)
    multiplier=torch.exp(2j*torch.pi*freq*fraction)[:,None]
    return torch.fft.ifft(torch.fft.fft(values,dim=0)*multiplier,dim=0).real


def rescale_for_action(del_residual,preconditioner,dt):
    return del_residual@preconditioner.T/dt**2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rest-cache',type=Path,required=True)
    parser.add_argument('--tangent-cache',type=Path,required=True)
    parser.add_argument('--dt',type=float,default=1/256)
    parser.add_argument('--period-updates',type=int,default=17564)
    parser.add_argument('--nodes',type=int,default=5)
    parser.add_argument('--axis',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--sign',type=int,choices=(-1,1),default=-1)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.nodes<3 or args.nodes%2!=1:raise ValueError('use odd temporal node count')
    rank.lock_inertia();torch.set_num_threads(2)
    dt=args.dt;n=args.period_updates;nodes=args.nodes
    rest=np.load(args.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64)
    F=np.load(args.tangent_cache)['following'];fac=lu_factor(F)
    # The resting exact following Jacobian preconditions DEL only. It does
    # not alter an equation, the endpoint, or the free evolution rule.
    preconditioner=torch.tensor(lu_solve(fac,np.eye(rest.size)),dtype=torch.float64)
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    clock_delta=torch.zeros((),dtype=torch.float64,requires_grad=True)
    independent=torch.nn.Parameter(ref.repeat(nodes-1,1))
    times=torch.arange(nodes,dtype=torch.float64)/nodes
    phase_fraction=torch.tensor(1/n,dtype=torch.float64)
    mask=rest.reshape(SIDE,SIDE,SIDE,12)
    vacuum=mask[0,0,0].copy();vacuum[:6]=0
    rest_contrast=ref-torch.tensor(np.tile(vacuum,SIDE**3),dtype=torch.float64)
    norm=torch.linalg.vector_norm(rest_contrast)

    def evaluate(intermediate,delta):
        periodic=torch.cat((ref[None,:],intermediate),dim=0)
        before=shift(time_shift(periodic,-phase_fraction),args.sign*(times-phase_fraction),args.axis)
        current=shift(periodic,args.sign*times,args.axis)
        after=shift(time_shift(periodic,phase_fraction),args.sign*(times+phase_fraction),args.axis)
        phase=theta+delta/n
        before=psi_clock(before,-phase);after=psi_clock(after,phase)
        # Historical action uses in-place accumulation of scalar energy;
        # evaluate each node independently instead of imposing vmap batching.
        del_values=torch.stack([action.del_residual(before[i],current[i],after[i],dt)
                                for i in range(nodes)])
        corrected=rescale_for_action(del_values,preconditioner,dt)
        return corrected,del_values,before,current,after

    def objective(backward=True):
        corrected,*_=evaluate(independent,clock_delta)
        loss=torch.sum(corrected**2)/(nodes*norm**2)
        if backward:loss.backward()
        return loss

    initial=float(torch.sqrt(objective(False)).detach())
    trace=[]
    optimizer=torch.optim.LBFGS([independent,clock_delta],lr=0.6,max_iter=args.iterations,
                  max_eval=2*args.iterations,tolerance_grad=1e-10,tolerance_change=1e-14,
                  line_search_fn='strong_wolfe',history_size=10)
    def closure():
        optimizer.zero_grad();loss=objective();trace.append(float(torch.sqrt(loss.detach())))
        if len(trace)%5==0:print('collocation evaluation',len(trace),'defect',trace[-1],flush=True)
        return loss
    optimizer.step(closure)
    c,raw,before,current,after=evaluate(independent,clock_delta)
    final=float(torch.linalg.vector_norm(c).detach()/(np.sqrt(nodes)*norm))
    phase=float(clock_delta.detach())
    result={'status':'V4_006_NATIVE_ACTION_RELATIVE_PERIODIC_COLLOCATION_SCREEN','action_sha256':sha256(HERE/'live_clifford_bond_action.py'),'tangent_cache_sha256':sha256(args.tangent_cache),'rest_cache_sha256':sha256(args.rest_cache),'dt':dt,'period_updates':n,'period_time':dt*n,'axis':args.axis,'sign':args.sign,'independent_time_nodes':nodes,'initial_acceleration_defect_over_contrast_norm':initial,'final_acceleration_defect_over_contrast_norm':final,'final_raw_DEL_RMS':float(torch.sqrt(torch.mean(raw**2)).detach()),'additional_clock_phase':phase,'optimizer_evaluations':len(trace),'trace':trace,'initial_state_fixed_to_exact_rest':True,'runtime_reset_or_target_force':False,'claim_boundary':'Sampled residual of an independent nonlinear boundary-value search. It is not a freely evolved orbit, nor a no-go theorem if it fails.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
