"""Separate the native handoff's static shape force from its clock/velocity.

Each term is varied from the same frozen action at the *same* sampled
nonlinear handoff shape. The co-rotating static triplet is a diagnostic, not
a prescribed fractional trajectory. Compute the full signed term response
using the incumbent following Jacobian as units, plus the exact sum check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_phase_corrected_handoff import native_layers, unpack
from v4_006_current_action_relative_orbit import psi_clock


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    state=np.load(a.state);nodes=torch.tensor(state['nodes'],dtype=torch.float64);dt=float(state['dt']);n=int(round(float(state['period_updates'])));phase=torch.tensor(float(state['phase_per_step']),dtype=nodes.dtype)
    axis=int(state['axis']);sign=int(state['sign']);ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=nodes.dtype);vac=ref.reshape(125,12)[0].clone();vac[:6]=0;norm=float(torch.linalg.vector_norm(ref-vac.repeat(125)))
    factor=lu_factor(np.load(a.tangent_cache)['following']);theta=torch.tensor(float(np.arccos(1-(dt*rank.R)**2/2)),dtype=nodes.dtype)
    terms={'base_and_common_material':lambda l,r:action.incumbent.interval(l,r,dt),'direct_prong_velocity':lambda l,r:dt*action.SCALE*action.one_form(l,r,dt),
           'six_face_oscillator':lambda l,r:action.SCALE*action.face_frame_oscillator_interval(l,r,dt),'spacetime_face':lambda l,r:action.SPACETIME_PLAQUETTE_SCALE*action.spacetime_factorized_face_interval(l,r,dt),
           'parallel_clock_completion':lambda l,r:action.PARALLEL_MATCHED_SCALE*action.parallel_overlap_matched_interval(l,r,dt)}
    times=torch.arange(nodes.shape[0],dtype=nodes.dtype)/nodes.shape[0];before,current,after=native_layers(nodes,times,n,phase,axis,sign)
    def scaled(v):return lu_solve(factor,v.detach().numpy())/dt**2
    def split(raw):
        p,c=unpack(torch.tensor(raw,dtype=nodes.dtype));rp,rc=unpack(ref);core=torch.sum(abs(rp)**2,dim=-1)>1e-12
        pp=float(torch.sum(abs(p)**2));cc=float(torch.sum(abs(c)**2));total=max(pp+cc,1e-300)
        return {'norm_over_contrast':float(np.linalg.norm(raw)/norm),'Psi_power_fraction':pp/total,'original_core_power_fraction':float(torch.sum(abs(p[core])**2+abs(c[core])**2))/total}
    rows=[]
    for i in range(nodes.shape[0]):
        if i==0:continue
        q=current[i];l=before[i];r=after[i];static_l=psi_clock(q[None,:],-theta)[0];static_r=psi_clock(q[None,:],theta)[0]
        raw=action.del_residual(l,q,r,dt);stat=action.del_residual(static_l,q,static_r,dt)
        d=scaled(raw);s=scaled(stat);pieces={};raw_sum=torch.zeros_like(raw)
        for label,fn in terms.items():
            def twin(x,y,z):return fn(x,y)+fn(y,z)
            grad=torch.func.grad(twin,argnums=1);val=grad(l,q,r);raw_sum=raw_sum+val;ds=scaled(val);ss=scaled(grad(static_l,q,static_r))
            pieces[label]={'dynamic':split(ds),'static':split(ss),'signed_dynamic_fraction':float(np.dot(d,ds)/max(np.dot(d,d),1e-300))}
        row={'fraction':float(times[i]),'dynamic':split(d),'static':split(s),'velocity_clock_difference':split(d-s),
             'static_dynamic_correlation':float(np.dot(d,s)/max(np.linalg.norm(d)*np.linalg.norm(s),1e-300)),
             'exact_term_sum_relative_error':float(torch.linalg.vector_norm(raw_sum-raw)/max(float(torch.linalg.vector_norm(raw)),1e-300)),'terms':pieces}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='terms'}),flush=True)
    result={'status':'V4_006_HANDOFF_FORCE_DECOMPOSITION_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),'state_sha256':sha256(a.state),'tangent_cache_sha256':sha256(a.tangent_cache),'dt':dt,'period_updates':n,'rows':rows,
            'claim_boundary':'On sampled native handoff seeds only. A large static component points to a shape/energy branch problem at this preparation; it is not a no-go for other discrete hopping trajectories.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
