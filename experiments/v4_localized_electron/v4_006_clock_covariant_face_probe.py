"""Exact tangent delta for the clock-charge-neutral face oscillator.

Let F(q) be the real representation of the complex face differences of
Psi and/or W/chi0, J=DF, and R(theta)=cos(theta)I+sin(theta)Omega. At the
exact co-rotating rest state, the old mixed interval Hessian is
-B/dt J^T R(theta) J and the replacement is -B/dt J^T J. Their DEL middle
block is fixed by the identical static profile restriction. This computes
the exact local reciprocal replacement without a new full tangent rebuild.
The candidate is never installed or credited with a prior stage pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import canonical_transfer_oscillator_spectrum as spectrum
import cube_incidence_extended_packet_gate as dirac
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_bond_action as incumbent
import relative_kinetic_probe as physical
import v4_006_clock_covariant_face_action as candidate
from v4_003_fixed_phase_space_readout import sha256
from v4_003_moving_native_graph_audit import HERE, sparse
from v4_006_one_cell_crossing import centered_target


def jacobian(rest,mode):
    psi,chi=physical.unpack_raw(rest,5);p=psi.reshape(125,3);c=chi.reshape(125,3)
    derivatives=np.zeros((125,3,12),complex);unit=np.column_stack((np.eye(3),1j*np.eye(3)))
    if mode=='Psi':derivatives[...,:6]=unit
    else:
        for j in range(6):
            derivatives[...,j]=np.cross(c,unit[:,j])/rank.R
            derivatives[...,j+6]=np.cross(unit[:,j],p)/rank.R
    idx=np.arange(125).reshape(5,5,5);r=[];col=[];values=[];row=0
    for axis in range(3):
        neighbor=np.roll(idx,-1,axis).ravel()
        for i,j in enumerate(neighbor):
            for site,block in ((i,-derivatives[i]),(j,derivatives[j])):
                block=np.vstack((block.real,block.imag));rr,cc=np.nonzero(block);r.extend(rr+row);col.extend(cc+12*site);values.extend(block[rr,cc])
            row+=6
    J=coo_matrix((values,(r,col)),shape=(row,rest.size)).tocsc()
    ids=np.arange(row).reshape(-1,6);rr=np.r_[ids[:,:3].ravel(),ids[:,3:].ravel()];cc=np.r_[ids[:,3:].ravel(),ids[:,:3].ravel()]
    omega=coo_matrix((np.r_[-np.ones(row//2),np.ones(row//2)],(rr,cc)),shape=(row,row)).tocsc()
    return J.T@J,J.T@omega@J


def delta(rest,mode,dt,theta):
    parts=('Psi','W') if mode=='both' else (mode,)
    mult=incumbent.SCALE/dt;Q,G=jacobian(rest,parts[0])
    if len(parts)==2:
        q,g=jacobian(rest,parts[1]);Q=Q+q;G=G+g
    F=-mult*((1-np.cos(theta))*Q-np.sin(theta)*G);P=F.T;C=-(F+P)
    return P.tocsc(),C.tocsc(),F.tocsc()


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--dt',type=float,default=1/256);p.add_argument('--cycles',type=float,default=100.);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt
    rest=np.load(a.rest_cache)['rest'];data=np.load(a.tangent_cache);theta=float(np.arccos(1-(dt*rank.R)**2/2));rotm=spectrum.rotation_matrix(rest.size,-theta);rotp=spectrum.rotation_matrix(rest.size,theta)
    P=sparse(data['previous']@rotm);C=sparse(data['current']);F=sparse(data['following']@rotp)
    psi_frame,chi_frame,sites=envelope.carrier_frames(rest);bases=envelope.flavor_bases()
    old_scales=json.loads((HERE/'v4_002_exact_1197/reach_100cycles_final.json').read_text())['chi_scales']
    probes=[];seeds_now=[];seeds_previous=[];target_one=[];target_end=[];target_end_i=[];steps=int(np.ceil(a.cycles*2*np.pi/(rank.R*dt)))
    for scale_kind in ('original_002','fixed_005'):
        for copy,basis in bases.items():
            alpha,beta=dirac.compressed_clifford(basis);chiscale=complex(old_scales[copy]['real'],old_scales[copy]['imag']) if scale_kind=='original_002' else 1/np.sqrt(rank.B)
            raw=envelope.raw_columns(basis,psi_frame,chi_frame,sites,dirac.OFFSETS,chiscale)
            for axis in range(4):
                for sign in (-1,1):
                    prev,now,after,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                    end=centered_target(now,alpha,beta,steps,dt,dirac.COMPARATOR_MASS)
                    seeds_previous.append(raw@dirac.real_coordinates(prev));seeds_now.append(raw@dirac.real_coordinates(now));target_one.append(raw@dirac.real_coordinates(after));target_end.append(raw@dirac.real_coordinates(end));target_end_i.append(raw@dirac.real_coordinates(1j*end))
                    probes.append({'preparation':scale_kind,'copy':copy,'axis':axis,'sign':sign})
    s0=np.column_stack(seeds_now);sm=np.column_stack(seeds_previous);tar1=np.column_stack(target_one);tend=np.column_stack(target_end);tend_i=np.column_stack(target_end_i)
    rv=np.random.default_rng(6006).normal(size=rest.size);rv/=np.linalg.norm(rv);layers=tuple(torch.tensor(x,dtype=torch.float64) for x in (rotm@rest,rest,rotp@rest))
    rows=[]
    for mode in ('baseline','Psi','W','both'):
        checks=[]
        if mode=='baseline':pp,cc,ff=P,C,F
        else:
            dp,dc,df=delta(rest,mode,dt,theta);pp,cc,ff=P+dp,C+dc,F+df
            def dd(p,q,r):
                def summ(l,m,n):return candidate.delta_interval(l,m,dt,mode)+candidate.delta_interval(m,n,dt,mode)
                return torch.func.grad(summ,argnums=1)(p,q,r)
            for j,mat,rotation in ((0,dp,rotm),(1,dc,np.eye(rest.size)),(2,df,rotp)):
                def fn(x):
                    q=list(layers);q[j]=x;return dd(*q)
                _,actual=torch.func.jvp(fn,(layers[j],),(torch.tensor(rotation@rv,dtype=torch.float64),))
                exact=mat@rv;checks.append(float(np.linalg.norm(actual.detach().numpy()-exact)/max(np.linalg.norm(exact),1e-300)))
            if max(checks)>1e-8:raise ValueError(f'derived DEL delta mismatch {mode}: {checks}')
        factor=splu(ff);advance=lambda n,m:factor.solve(-(cc@n+pp@m));now=s0.copy();prev=sm.copy();first=advance(now,prev)
        initial=np.sqrt(np.sum(now**2+prev**2,axis=0));vel_initial=np.sqrt(np.sum(now**2+((now-prev)/dt)**2,axis=0));max_ratio=np.ones(now.shape[1]);max_vel=np.ones(now.shape[1]);stopped=False
        for step in range(steps):
            nxt=advance(now,prev);prev,now=now,nxt
            ratios=np.sqrt(np.sum(now**2+prev**2,axis=0))/initial;vel=np.sqrt(np.sum(now**2+((now-prev)/dt)**2,axis=0))/vel_initial
            max_ratio=np.maximum(max_ratio,ratios);max_vel=np.maximum(max_vel,vel)
            if not np.isfinite(ratios).all() or max(max_ratio)>1e8:stopped=True;break
        percase=[]
        for j,probe in enumerate(probes):
            aligned=np.column_stack((tend[:,j],tend_i[:,j]));fit=np.linalg.lstsq(aligned,now[:,j],rcond=1e-12)[0];aligned=aligned@(fit/max(np.linalg.norm(fit),1e-300))
            percase.append(dict(probe,one_step_state_error=float(np.linalg.norm(first[:,j]-tar1[:,j])/np.linalg.norm(tar1[:,j])),one_step_acceleration_error=float(np.linalg.norm(first[:,j]-tar1[:,j])/np.linalg.norm(tar1[:,j])/dt**2),max_norm_ratio=float(max_ratio[j]),max_velocity_norm_ratio=float(max_vel[j]),final_Dirac_error=None if stopped else float(np.linalg.norm(now[:,j]-aligned)/np.linalg.norm(tend[:,j]))))
        row={'mode':mode,'delta_jvp_relative_errors':checks,'steps_completed':step+1,'stopped_due_to_growth':stopped,'cases':percase,'summaries':{}}
        for kind in ('original_002','fixed_005'):
            subset=[r for r in percase if r['preparation']==kind];row['summaries'][kind]={k:max(r[k] for r in subset) for k in ('one_step_state_error','one_step_acceleration_error','max_norm_ratio','max_velocity_norm_ratio')}
            if not stopped:row['summaries'][kind]['final_Dirac_error']=max(r['final_Dirac_error'] for r in subset)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
    result={'status':'V4_006_CLOCK_COVARIANT_FACE_REPLACEMENT_PROBE_NOT_ADMITTED','incumbent_action_sha256':sha256(Path(incumbent.__file__)),'candidate_sha256':sha256(Path(candidate.__file__)),
            'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),'tangent_cache_sha256':sha256(a.tangent_cache),'dt':dt,'cycles':a.cycles,'steps':steps,'rows':rows,
            'claim_boundary':'Exact change to full native DEL tangent, independently JVP checked. Action variants are research replacements, no nonlinear or movement pass is inferred.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
