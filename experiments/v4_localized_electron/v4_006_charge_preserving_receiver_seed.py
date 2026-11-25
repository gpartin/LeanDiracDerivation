"""Derive the initial receiver phase and chi tilt from native face charges.

With the original eight-site core exactly unchanged, a weak receiver Psi
signal changes the face charge norm P by a linear interference plus a
positive quadratic term. Solve the two possible phase offsets exactly.
The W norm then determines the minimum chi orientation correction along its
native receiver-shell gradient. Both are algebraic consequences of the
existing six-face action, not receiver-setting rules or runtime forces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import root

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_reduced_handoff import rotation
from v4_006_face_charge_transport_seed import bond_norm,pack,unpack


def bond_inner(x,y):
    return sum(torch.sum((torch.roll(x,-1,a)-x).conj()*(torch.roll(y,-1,a)-y)) for a in range(3))


def geometric_seed(p,c,tail,mask,epsilon,branch):
    overlap=complex(bond_inner(p,tail));norm_tail=float(bond_norm(tail));target=float(bond_norm(torch.linalg.cross(c,p,dim=-1)))
    cosine=-epsilon*norm_tail/(2*abs(overlap));phi=branch*np.arccos(cosine)-np.angle(overlap)
    pn=p+epsilon*np.exp(1j*phi)*tail
    # W quadratic in the exterior chi change. Its exact receiver-supported
    # steepest charge descent is independent of the target tilt assignment.
    def Wnorm(cv):return bond_norm(torch.linalg.cross(cv,pn,dim=-1))
    grad=torch.func.grad(Wnorm)(c)*mask[...,None];direction=-grad/torch.linalg.vector_norm(grad)
    base=torch.linalg.cross(c,pn,dim=-1);variation=torch.linalg.cross(direction,pn,dim=-1)
    aa=float(bond_norm(variation));bb=2*float(bond_inner(base,variation).real);cc=float(bond_norm(base))-target
    disc=bb*bb-4*aa*cc
    if disc<0:raise ValueError(f'no real receiver tilt at epsilon={epsilon}, branch={branch}')
    roots=np.roots([aa,bb,cc]);b=min((float(x.real) for x in roots if abs(x.imag)<1e-10),key=abs)
    return pn,c+b*direction,phi,b,direction


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--dt',type=float,default=1/256)
    p.add_argument('--axis',type=int,default=0);p.add_argument('--sign',type=int,default=-1);p.add_argument('--amplitudes',type=float,nargs='+',default=[.001,.01,.05]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt;ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=torch.float64);pr,cr=unpack(ref)
    core=torch.sum(abs(pr)**2,dim=-1)>1e-12;receiver=torch.roll(core,a.sign,a.axis)&~core;tail=torch.roll(pr,a.sign,a.axis)*receiver[...,None]
    target_tilt=(torch.roll(cr,a.sign,a.axis)-cr)*receiver[...,None];theta=float(np.arccos(1-(dt*rank.R)**2/2));phase=torch.tensor([theta,0.],dtype=ref.dtype)
    def ell(q,ph):return action.interval(q,rotation(q,ph),dt)
    qmap=torch.func.grad(ell,argnums=1);Q=qmap(ref,phase).detach().numpy();scale=max(abs(Q[0]),1.)
    rows=[]
    for amp in a.amplitudes:
        for branch in (-1,1):
            pn,cn,phi,tilt,direction=geometric_seed(pr,cr,tail,receiver,amp,branch)
            qg=pack(pn,cn);Qg=qmap(qg,phase).detach().numpy()
            # Free phase and receiver tilt amplitudes refine the two exact
            # conserved action charges, not a Dirac endpoint value.
            def prep(v):return pack(pr+amp*np.exp(1j*float(v[0]))*tail,cr+float(v[1])*direction)
            def fun(v):return (qmap(prep(v),phase).detach().numpy()-Q)/scale
            solved=root(fun,[phi,tilt],method='hybr',options={'xtol':1e-10,'maxfev':30});q=prep(solved.x);pn,cn=unpack(q)
            # The freely solved orientation is compared to the receiving
            # cube's shifted frame; no such frame was used to compute grad.
            dc=cn-cr;alignment=complex(torch.sum(target_tilt.conj()*dc));denom=float(torch.linalg.vector_norm(target_tilt)*torch.linalg.vector_norm(dc))
            result={'Psi_receiver_amplitude_fraction':amp,'phase_branch':branch,'derived_geometric_phase':float(phi),'refined_native_phase':float(solved.x[0]),
                    'geometric_chi_tilt_norm':float(abs(tilt)),'refined_chi_tilt_norm':float(abs(solved.x[1])),
                    'geometric_charge_defect':float(np.linalg.norm((Qg-Q)/scale)),'refined_charge_defect':float(np.linalg.norm(fun(solved.x))),
                    'refined_core_difference':float(torch.linalg.vector_norm((pn-pr)[core])+torch.linalg.vector_norm((cn-cr)[core])),
                    'chi_receiver_tilt_relative_to_full_target':float(torch.linalg.vector_norm(dc)/torch.linalg.vector_norm(target_tilt)),
                    'chi_tilt_alignment_with_translated_target_abs':abs(alignment)/max(denom,1e-300),'chi_tilt_alignment_with_translated_target_real':alignment.real/max(denom,1e-300),
                    'root_evaluations':int(solved.nfev)}
            rows.append(result);print(json.dumps(result),flush=True)
    result={'status':'V4_006_CHARGE_PRESERVING_RECEIVER_PREPARATION_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),'rest_cache_sha256':sha256(a.rest_cache),
            'dt':dt,'axis':a.axis,'sign':a.sign,'receiver_sites':int(receiver.sum()),'rest_phase_charge':Q.tolist(),'rows':rows,
            'claim_boundary':'Two conserved charges and exact original core are respected by candidate preparatory states. This does not show the receiving state develops dynamically, nor close a cell handoff.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
