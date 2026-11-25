"""Unrestricted first pushes of the charge-derived native receiver precursor.

Only the two initial layers are prepared by the exact geometric charge
constraints. All subsequent layers solve the original implicit DEL. The
signed receiver-amplitude rate is a seed variable, not a force or reset.
This is a short nonlinear test of the mechanism, not a full crossing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator,gmres

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_preserving_receiver_seed import geometric_seed
from v4_006_charge_reduced_handoff import rotation
from v4_006_face_charge_transport_seed import pack,unpack
from v4_006_phase_corrected_handoff import charge,reduced_del


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--dt',type=float,default=1/256);p.add_argument('--axis',type=int,default=0);p.add_argument('--sign',type=int,default=-1)
    p.add_argument('--amplitude',type=float,default=.01);p.add_argument('--period-updates',type=int,default=17564);p.add_argument('--updates',type=int,default=4);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matrix-free-newton',action='store_true',help='refresh the exact candidate-layer Jacobian if the rest preconditioner stalls')
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt;ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=torch.float64);pr,cr=unpack(ref)
    core=torch.sum(abs(pr)**2,dim=-1)>1e-12;receiver=torch.roll(core,a.sign,a.axis)&~core;tail=torch.roll(pr,a.sign,a.axis)*receiver[...,None]
    target_tilt=(torch.roll(cr,a.sign,a.axis)-cr)*receiver[...,None];theta=float(np.arccos(1-(dt*rank.R)**2/2));phase=torch.tensor([theta,0.],dtype=ref.dtype)
    Q=charge(rotation(ref,-phase),ref,dt).detach().numpy();qs=max(abs(Q[0]),1.);factor=lu_factor(np.load(a.tangent_cache)['following']);norm=float(torch.linalg.vector_norm(pr));tn=float(torch.sum(abs(tail)**2));rows=[]
    def qstatic(q,ph):return action.interval(q,rotation(q,ph),dt)
    qmap=torch.func.grad(qstatic,argnums=1)
    def seed(eps,branch):
        pn,cn,phi,tilt,direction=geometric_seed(pr,cr,tail,receiver,eps,branch)
        def prep(v):return pack(pr+eps*np.exp(1j*float(v[0]))*tail,cr+float(v[1])*direction)
        def fun(v):return (qmap(prep(v),phase).detach().numpy()-Q)/qs
        solved=root(fun,[phi,tilt],method='hybr',options={'xtol':1e-10,'maxfev':30})
        if np.linalg.norm(fun(solved.x))>1e-10:raise ValueError('static native charge seed failed')
        return prep(solved.x),float(solved.x[0]),direction
    def native_phase(l,r):
        def fun(v):return (charge(l,rotation(r,torch.tensor(v,dtype=ref.dtype)),dt).detach().numpy()-Q)/qs
        opt=root(fun,phase.numpy(),method='hybr',options={'xtol':1e-10,'maxfev':40});res=float(np.linalg.norm(fun(opt.x)))
        if res>1e-10:raise ValueError('incoming charge solve failed')
        return torch.tensor(opt.x,dtype=ref.dtype),res
    def solve(p,q,f):
        # Solve actual layer equation; F is a numerical preconditioner only.
        history=[]
        def residual(x):return lu_solve(factor,reduced_del(p,q,x,dt).detach().numpy())
        for _ in range(18 if not a.matrix_free_newton else 5):
            r=residual(f);err=float(np.linalg.norm(r)/norm/dt**2);history.append(err)
            if err<1e-8:break
            accepted=False
            for k in range(10):
                trial=f-torch.tensor(2.0**(-k)*r,dtype=ref.dtype)
                if np.linalg.norm(residual(trial))<np.linalg.norm(r):f=trial;accepted=True;break
            if not accepted:break
        if a.matrix_free_newton:
            for _ in range(3):
                r=residual(f);err=float(np.linalg.norm(r)/norm/dt**2);history.append(err)
                if err<1e-8:break
                def matvec(v):
                    _,jvp=torch.func.jvp(lambda x:reduced_del(p,q,x,dt),(f,),(torch.tensor(v,dtype=ref.dtype),))
                    return lu_solve(factor,jvp.detach().numpy())
                operator=LinearOperator((f.numel(),)*2,matvec=matvec,dtype=np.float64)
                correction,info=gmres(operator,-r,rtol=1e-4,atol=0.,restart=20,maxiter=2)
                if info<0:break
                accepted=False
                for k in range(10):
                    trial=f+torch.tensor(2.0**(-k)*correction,dtype=ref.dtype)
                    if np.linalg.norm(residual(trial))<np.linalg.norm(r):f=trial;accepted=True;break
                if not accepted:break
        exact=action.del_residual(p,q,f,dt).detach().numpy();error=float(np.linalg.norm(lu_solve(factor,exact))/norm/dt**2)
        return f,history,error
    delta=1/a.period_updates
    for branch in (-1,1):
        q,phi,tilt=seed(a.amplitude,branch);p0,_,_=seed(a.amplitude-delta,branch)
        native_in,in_def=native_phase(p0,q);prev=rotation(p0,-native_in);now=q.clone();samples=[]
        # No runtime alignment/projection: global clock removal below is
        # used solely for measurements. The native tensors are unmodified.
        for step in range(a.updates+1):
            np_,nc=unpack(now);clock=torch.angle(torch.sum(pr[core].conj()*np_[core]));pa=np_*torch.exp(-1j*clock);z=complex(torch.sum(tail.conj()*pa)/tn)
            chi_m=nc-cr;ct=float(torch.linalg.vector_norm(chi_m[receiver]));receiver_direction=complex(torch.sum(target_tilt.conj()*chi_m))/max(float(torch.linalg.vector_norm(target_tilt))*float(torch.linalg.vector_norm(chi_m)),1e-300)
            qnative=charge(prev,now,dt).detach().numpy();initial_charge_error=((qnative-Q)/qs).tolist()
            row={'step':step,'receiver_Psi_coefficient':[z.real,z.imag],'receiver_signed_initial_phase_amplitude':float(np.real(z*np.exp(-1j*phi))),
                 'receiver_Psi_amplitude':abs(z),'chi_receiver_tilt_norm':ct,'chi_receiver_alignment_to_shifted_frame_real':receiver_direction.real,
                 'chi_receiver_alignment_to_shifted_frame_abs':abs(receiver_direction),'charge_relative_defect':initial_charge_error,
                 'Psi_core_fidelity':float(abs(torch.sum(pr.conj()*pa[...]))**2/(torch.sum(abs(pr)**2)*torch.sum(abs(pa)**2)))}
            samples.append(row)
            if step==a.updates:break
            # Rest-clock extrapolation is a numerical guess, not a target.
            guess=rotation(2*now-rotation(prev,phase),phase)
            following,history,error=solve(prev,now,guess);row['next_layer_DEL_scaled_error']=error;row['next_layer_solver_history']=history
            if error>=1e-8:break
            prev,now=now,following
        first,last=samples[0],samples[-1];r={'phase_branch':branch,'initial_native_phase':native_in.tolist(),'incoming_charge_root_defect':in_def,'initial_receiver_phase':phi,
             'steps_completed':len(samples)-1,'samples':samples,'receiver_amplitude_change':last['receiver_signed_initial_phase_amplitude']-first['receiver_signed_initial_phase_amplitude'],
             'receiver_amplitude_expected_seed_change':(len(samples)-1)*delta,'claim':'unrestricted native evolution after two prepared layers'}
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='samples'}),flush=True)
    result={'status':'V4_006_CHARGE_DERIVED_RECEIVER_FREE_RELEASE_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),'rest_cache_sha256':sha256(a.rest_cache),'tangent_cache_sha256':sha256(a.tangent_cache),
            'dt':dt,'axis':a.axis,'sign':a.sign,'initial_seed_amplitude':a.amplitude,'initial_seed_rate_per_step':delta,'matrix_free_newton':a.matrix_free_newton,'rows':rows,'claim_boundary':'Native nonlinear first pushes on charge-derived precursor; no runtime projection or Dirac force. This is not a complete cell handoff or proof that the precursor forms from quiet vacuum.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
