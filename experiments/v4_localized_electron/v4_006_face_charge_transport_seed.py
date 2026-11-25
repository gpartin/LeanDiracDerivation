"""Native charge-preserving handoff seed, derived from face geometry.

The large B-normalized six-face oscillator contributes phase momenta
 Qpsi=B sin(theta)/dt * (||D_face Psi||^2+||D_face W||^2/chi0^2),
 Qchi=B sin(theta)/dt * ||D_face W||^2/chi0^2 at a static chi/common Psi clock.
For a candidate intermediate shape, keep these two bond norms by first
solving the Psi scale, then the quadratic in the chi contrast scale. Refine
both amplitudes against the full exact native phase-charge equations. This
is a preparation for a boundary solve, not an imposed evolution rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import root

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_reduced_handoff import rotation
from v4_006_current_action_relative_orbit import shift


def unpack(q):
    g=q.reshape(5,5,5,4,3)
    return torch.complex(g[...,0,:],g[...,1,:]),torch.complex(g[...,2,:],g[...,3,:])


def pack(p,c):return torch.stack((p.real,p.imag,c.real,c.imag),dim=-2).reshape(-1)


def bond_norm(q):
    return sum(torch.sum(abs(torch.roll(q,-1,axis)-q)**2) for axis in range(3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--dt',type=float,default=1/256)
    p.add_argument('--period-updates',type=int,default=17564);p.add_argument('--nodes',type=int,default=5);p.add_argument('--axis',type=int,default=0);p.add_argument('--sign',type=int,default=-1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--state-output',type=Path,required=True);a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64);psi0,chi0=unpack(ref);vac=chi0[0,0,0].clone();dc0=chi0-vac
    psi1=torch.roll(psi0,a.sign,a.axis);dc1=torch.roll(dc0,a.sign,a.axis);dt=a.dt;theta=float(np.arccos(1-(dt*rank.R)**2/2));phase=torch.tensor([theta,0.],dtype=ref.dtype)
    def reduced(q,ph):return action.interval(q,rotation(q,ph),dt)
    charge=torch.func.grad(reduced,argnums=1);Q=charge(ref,phase).detach().numpy();scl=np.maximum(abs(Q),1.)
    Ptarget=float(bond_norm(psi0));Wtarget=float(bond_norm(torch.linalg.cross(chi0,psi0,dim=-1)))
    prefactor=action.SCALE*np.sin(theta)/dt
    formula=np.array([prefactor*(Ptarget+Wtarget/rank.R**2),prefactor*Wtarget/rank.R**2]);
    def face(q,ph):return action.SCALE*action.face_frame_oscillator_interval(q,rotation(q,ph),dt)
    faceQ=torch.func.grad(face,argnums=1)(ref,phase).detach().numpy()
    ref_r=-(float(reduced(ref,phase))-theta*Q[0])/dt
    rows=[];native_nodes=[];times=torch.arange(a.nodes,dtype=ref.dtype)/a.nodes
    for t in times:
        f=float((1-torch.cos(torch.pi*t))/2)
        p0=(1-f)*psi0+f*psi1;dc=(1-f)*dc0+f*dc1
        a0=np.sqrt(Ptarget/float(bond_norm(p0)));p_now=a0*p0
        const=torch.linalg.cross(vac.expand_as(p_now),p_now,dim=-1);linear=torch.linalg.cross(dc,p_now,dim=-1)
        aa=float(bond_norm(linear));cc=float(bond_norm(const));bb=float(bond_norm(const+linear))-aa-cc
        roots=np.roots([aa,bb,cc-Wtarget]);real=[float(r.real) for r in roots if abs(r.imag)<1e-10 and r.real>0]
        if not real:raise ValueError(f'no positive geometric correction at {f}: {roots}')
        b0=min(real,key=lambda x:abs(x-1))
        def profile(x):return pack(float(x[0])*p0,vac+float(x[1])*dc)
        def fun(x):return (charge(profile(x),phase).detach().numpy()-Q)/scl
        ans=root(fun,[a0,b0],method='hybr',options={'xtol':1e-10,'maxfev':35});q=profile(ans.x);native_nodes.append(q)
        u,v=unpack(q);Qnow=charge(q,phase).detach().numpy()
        row={'time_fraction':float(t),'transfer_seed_fraction':f,'geometric_Psi_scale':a0,'geometric_chi_contrast_scale':b0,
             'exact_charge_Psi_scale':float(ans.x[0]),'exact_charge_chi_contrast_scale':float(ans.x[1]),
             'geometric_seed_charge_defect':float(np.linalg.norm(fun([a0,b0]))),'full_charge_defect':float(np.linalg.norm((Qnow-Q)/scl)),
             'Psi_norm_ratio':float(torch.linalg.vector_norm(u)/torch.linalg.vector_norm(psi0)),
             'chi_contrast_norm_ratio':float(torch.linalg.vector_norm(v-vac)/torch.linalg.vector_norm(dc0)),
             'face_Psi_norm_ratio':float(bond_norm(u)/Ptarget),'face_W_norm_ratio':float(bond_norm(torch.linalg.cross(v,u,dim=-1))/Wtarget),
             'fixed_charge_Routhian_delta':-(float(reduced(q,phase))-theta*Q[0])/dt-ref_r,
             'phase_roots_evaluations':int(ans.nfev)}
        rows.append(row);print(json.dumps(row),flush=True)
    # Seed can be handed to the free-period or phase-momentum-elimination
    # programs. The native equations never contain the scaling path.
    nodes=shift(torch.stack(native_nodes),-a.sign*times,a.axis).detach().numpy()
    np.savez_compressed(a.state_output,nodes=nodes,period_updates=a.period_updates,phase_per_step=theta,dt=dt,axis=a.axis,sign=a.sign)
    result={'status':'V4_006_NATIVE_FACE_CHARGE_PREPARATION','action_sha256':sha256(Path(action.__file__)),'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),
            'dt':dt,'axis':a.axis,'sign':a.sign,'nodes':a.nodes,'phase_charge_rest':Q.tolist(),
            'face_charge_formula':formula.tolist(),'face_charge_ad':faceQ.tolist(),'face_charge_formula_relative_defect':float(np.linalg.norm(formula-faceQ)/np.linalg.norm(faceQ)),
            'fixed_charge_routhian_rest':ref_r,'rows':rows,'state_sha256':sha256(a.state_output),
            'claim_boundary':'Static phase-momentum-preserving seed through geometric native charge constraints. No unrestricted moving trajectory, energy gate, or Dirac motion pass is inferred.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
