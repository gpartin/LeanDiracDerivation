"""Derive and test the oscillator's local phase-momentum transfer rank.

At F_±=exp(±i theta)(F±h u), let L be the positive six-face Laplacian,
R_i=Re(F_i† L u_i), S_i=Re(u_i† L F_i), K=B sin(theta)/h. To first order,
the time rate of native charge storage is K(R+S), the face inflow is
K(S-R), and the local imbalance is 2K R. Apply these identities to Psi
and W=chi wedge Psi/chi0 (W feeds both conserved charges). Direct chi
has theta=0 and contributes no first-order rate on this centered chart.

Solve the *inverse* local momentum requirement for fixed charge redistribution
rate; report phase-only versus full joint velocity capacity, leakage and
response cost. Neither solution is inserted in the action or used as a
runtime correction. Finite handoffs need unrestricted DEL and free release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve,svd

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_reduced_handoff import rotation
from v4_006_face_phase_momentum import face_charge,divergence,local_contract
from v4_006_phase_corrected_handoff import project_face_capacity


SIDE=5;SITES=SIDE**3


def unpack(raw):
    q=np.asarray(raw).reshape(SIDE,SIDE,SIDE,4,3)
    return q[...,0,:]+1j*q[...,1,:],q[...,2,:]+1j*q[...,3,:]


def pack(p,c):return np.stack((p.real,p.imag,c.real,c.imag),axis=-2).reshape(-1)


def L(f):return 6*f-sum(np.roll(f,sign,axis) for axis in range(3) for sign in (-1,1))


def maps(raw):
    """Rows: one local Psi and one local chi phase charge; columns qdot."""
    p,c=unpack(raw);w=np.cross(c,p)/rank.R;size=raw.size
    # Exact real-linear derivative of each native complex frame field.
    eye=np.eye(size).reshape(SIDE,SIDE,SIDE,4,3,size)
    up=eye[...,0,:,:]+1j*eye[...,1,:,:];uc=eye[...,2,:,:]+1j*eye[...,3,:,:]
    uw=(np.cross(c[...,None],up,axisa=3,axisb=3,axisc=3)+np.cross(uc,p[...,None],axisa=3,axisb=3,axisc=3))/rank.R
    def part(f,u):
        r=np.real(np.sum(f.conj()[...,None]*L(u),axis=3)).reshape(SITES,size)
        s=np.real(np.sum(u.conj()*L(f)[...,None],axis=3)).reshape(SITES,size)
        return r+s,s-r,2*r
    qp,jp,rp=part(p,up);qw,jw,rw=part(w,uw)
    storage=np.vstack((qp+qw,qw));inflow=np.vstack((jp+jw,jw));imbalance=np.vstack((rp+rw,rw))
    # Local phase adjustments: one angular rate per native field and site.
    cols=[]
    for site in range(SITES):
        index=np.unravel_index(site,(SIDE,)*3)
        for family in (0,1):
            pp=np.zeros_like(p);cc=np.zeros_like(c)
            if family==0:pp[index]=1j*p[index]
            else:cc[index]=1j*c[index]
            v=pack(pp,cc);n=np.linalg.norm(v)
            if n>1e-12:cols.append(v/n)
    phase=np.column_stack(cols)
    return storage,inflow,imbalance,phase


def solve_rate(storage,inflow,u,phase=None):
    # Demand BOTH the same local charge-storage rate and matching inflow.
    target=storage@u;rhs=np.r_[np.zeros_like(target),target-inflow@u]
    operator=np.vstack((storage,inflow))
    if phase is not None:operator=operator@phase
    u0,s,vh=svd(operator,full_matrices=False);rankn=int(np.count_nonzero(s>s[0]*1e-10)) if s.size and s[0]>0 else 0
    correction=vh[:rankn].T@((u0[:,:rankn].T@rhs)/s[:rankn]) if rankn else np.zeros(operator.shape[1])
    if phase is not None:correction=phase@correction
    fixed=u+correction;nr=max(np.linalg.norm(target),1e-300)
    return fixed,{'rank':rankn,'raw_columns':int(operator.shape[1]),'storage_rate_relative_defect':float(np.linalg.norm(storage@fixed-target)/nr),
                 'inflow_rate_relative_defect':float(np.linalg.norm(inflow@fixed-target)/nr),
                 'local_balance_relative_defect':float(np.linalg.norm((storage-inflow)@fixed)/nr),
                 'velocity_correction_norm_ratio':float(np.linalg.norm(correction)/max(np.linalg.norm(u),1e-300)),
                 'corrected_velocity_norm_ratio':float(np.linalg.norm(fixed)/max(np.linalg.norm(u),1e-300)),
                 'retained_original_velocity_component':float(np.dot(fixed,u)/max(np.dot(u,u),1e-300))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rest-cache',type=Path,required=True);ap.add_argument('--tangent-cache',type=Path,required=True)
    ap.add_argument('--dt',type=float,default=1/256);ap.add_argument('--period-updates',type=int,default=17564);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--axes',nargs='+',type=int,default=(0,1,2));ap.add_argument('--endpoint-phase',type=float,default=0.)
    a=ap.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64);dt=a.dt;T=a.period_updates*dt;theta=float(np.arccos(1-(dt*rank.R)**2/2));K=rank.B*np.sin(theta)/dt
    factor=lu_factor(np.load(a.tangent_cache)['following']);p0,c0=unpack(rest);core=np.linalg.norm(p0,axis=-1)>1e-12;vac=np.zeros_like(rest.reshape(SIDE,SIDE,SIDE,12));vac[...,8]=rank.R
    norm=np.linalg.norm(rest-vac.reshape(-1));rows=[]
    phase=torch.tensor([theta,0.],dtype=ref.dtype)
    def layers(q,u,mult=1.):
        l=rotation(torch.tensor(q-dt*mult*u,dtype=ref.dtype),-phase).numpy();r=rotation(torch.tensor(q+dt*mult*u,dtype=ref.dtype),phase).numpy()
        return l,q,r
    def face_balance(q,u,eps):
        l,m,r=layers(q,u,eps);bp,_=face_charge(l,m,dt);bf,_=face_charge(m,r,dt)
        return ((bf[1]-bp[1])-divergence(bf[2])).reshape(-1)/dt
    def full(q,u):
        l,m,r=layers(q,u);g=action.del_residual(torch.tensor(l),torch.tensor(m),torch.tensor(r),dt).detach().numpy()
        return {'full_DEL_acceleration_defect':float(np.linalg.norm(lu_solve(factor,g))/dt**2/norm),
                'full_local_phase_defect_rate_over_K':float(np.linalg.norm(local_contract(m,g))/dt/K)}
    for axis in a.axes:
        for sign in (-1,1):
            translated=torch.roll(ref.reshape(SIDE,SIDE,SIDE,12),sign,axis).reshape(-1)
            translated=rotation(translated,torch.tensor([a.endpoint_phase,0.],dtype=ref.dtype))
            receiver=np.roll(core,sign,axis)&~core;sender=core&~np.roll(core,sign,axis)
            def shape(t):
                f=(1-np.cos(np.pi*t))/2
                return project_face_capacity(((1-f)*ref+f*translated)[None,:],ref).detach().numpy()[0]
            for t in (.25,.5,.75):
                q=shape(t);h=1e-5;u=(shape(t+h)-shape(t-h))/(2*h*T)
                storage,inflow,imbalance,phase_basis=maps(q);g=storage@u;nr=max(np.linalg.norm(g),1e-300)
                pair=face_balance(q,u,1e-3)-face_balance(q,u,-1e-3)
                exact=pair/(2e-3*K);theory=imbalance@u
                up,phase_result=solve_rate(storage,inflow,u,phase_basis);uf,full_result=solve_rate(storage,inflow,u)
                def gain(v):return (K*(inflow@v).reshape(2,SIDE,SIDE,SIDE)[:,receiver].sum(axis=1)).tolist()
                baseline={'local_balance_relative_defect':float(np.linalg.norm(theory)/nr),'receiver_required_rate':(K*g.reshape(2,SIDE,SIDE,SIDE)[:,receiver].sum(axis=1)).tolist(),
                          'receiver_actual_rate':gain(u)}
                phase_result.update(receiver_inflow_rate=gain(up));full_result.update(receiver_inflow_rate=gain(uf))
                # Native full DEL on the middle finite handoff shape. This is
                # a diagnostic of a local solution of the phase subsystem.
                if t==.5:
                    baseline.update(full(q,u));phase_result.update(full(q,up));full_result.update(full(q,uf))
                row={'axis':axis,'sign':sign,'fraction':t,'phase_imbalance_formula_relative_defect':float(np.linalg.norm(exact-theory)/max(np.linalg.norm(theory),1e-300)),
                     'matrix_identity_defect':float(np.linalg.norm(storage-inflow-imbalance)/max(np.linalg.norm(storage),1e-300)),
                     'baseline':baseline,'phase_only_correction':phase_result,'full_velocity_correction':full_result}
                rows.append(row);print(axis,sign,t,'baseline',baseline['local_balance_relative_defect'],'phase',phase_result['local_balance_relative_defect'],'joint',full_result['local_balance_relative_defect'],flush=True)
    result={'status':'V4_006_LOCAL_PHASE_MOMENTUM_INVERSE_REQUIREMENT','action_sha256':sha256(Path(action.__file__)),'rest_cache_sha256':sha256(a.rest_cache),'tangent_cache_sha256':sha256(a.tangent_cache),
            'dt':dt,'period_updates':a.period_updates,'endpoint_Psi_phase':a.endpoint_phase,'phase_rate_prefactor':K,'rows':rows,
            'claim_boundary':'Exact finite-layer oscillator first variation and numerical rank of a local inverse velocity requirement on defined charge-capacity seeds. This is not integrated motion or a new action rule.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
