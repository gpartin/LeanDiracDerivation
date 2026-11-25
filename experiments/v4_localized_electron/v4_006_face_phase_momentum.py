"""Exact local Noether phase momentum of the admitted six-face oscillator.

Each positive face edge i->j is counted once. If d=F_j-F_i, the matched
discrete oscillator interval is L_e=M/(2h)[c(|d_l|²+|d_r|²)-2Re(d_l†d_r)],
where c=cos(theta). Its exact endpoint momentum at j is M/h*(c*d-d_other),
and at i its negative. Contraction with iF gives Q^- and Q^+ and their
sum is the signed inflow I_ij to j. W=chi wedge Psi transforms with the sum
of both native phases, so W contributes to both currents. Direct chi has
c=1 and M=B². No new action term, projected update, or fitted phase rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_current_action_relative_orbit import psi_clock
from v4_006_phase_corrected_handoff import native_layers


def unpack(raw):
    g=np.asarray(raw).reshape(5,5,5,4,3)
    return g[...,0,:]+1j*g[...,1,:],g[...,2,:]+1j*g[...,3,:]


def dotimag(x,y):return np.imag(np.sum(x.conj()*y,axis=-1))


def one_field(left,right,dt,coefficient,cosine):
    ql=np.zeros(left.shape[:-1]);qr=ql.copy();flux=[];midpoint_flux=[]
    for axis in range(3):
        lj=np.roll(left,-1,axis);rj=np.roll(right,-1,axis);dl=lj-left;dr=rj-right
        ml=coefficient/dt*(cosine*dl-dr);mr=coefficient/dt*(cosine*dr-dl)
        left_i=-dotimag(left,ml);left_j=dotimag(lj,ml)
        right_i=-dotimag(right,mr);right_j=dotimag(rj,mr)
        ql+=left_i+np.roll(left_j,1,axis);qr+=right_i+np.roll(right_j,1,axis)
        torque=left_j+right_j;flux.append(torque)
        # Independent midpoint algebra: I_ij = M*h*[omega0² Im(Fj†Fi)
        # - (1+c)/2 Im(dFj†dFi)]. This formula applies to chi at omega0=0.
        a=(1+cosine)/2;omega0sq=2*(1-cosine)/dt**2
        mid=(left+right)/2;vel=(right-left)/dt
        midj=np.roll(mid,-1,axis);velj=np.roll(vel,-1,axis)
        midpoint_flux.append(coefficient*dt*(omega0sq*dotimag(midj,mid)-a*dotimag(velj,vel)))
    return ql,qr,np.asarray(flux),np.asarray(midpoint_flux)


def face_charge(left,right,dt):
    lp,lc=unpack(left);rp,rc=unpack(right);clock=rank.R;B=rank.B
    cos=1-dt**2*clock**2/2
    p=one_field(lp,rp,dt,B,cos)
    w=one_field(np.cross(lc,lp)/clock,np.cross(rc,rp)/clock,dt,B,cos)
    c=one_field(lc,rc,dt,B**2,1.)
    combined=tuple(np.stack((p[k]+w[k],c[k]+w[k])) for k in range(4))
    return combined,dict(Psi=p,W=w,chi=c)


def divergence(flux):return sum(np.roll(flux[:,a],1,a+1)-flux[:,a] for a in range(3))


def local_contract(q,grad):
    p,c=unpack(q);gp,gc=unpack(grad)
    return np.stack((dotimag(p,gp),dotimag(c,gc)))


def ad_charge(fn,left,right,dt):
    l=torch.tensor(left,dtype=torch.float64);r=torch.tensor(right,dtype=torch.float64)
    gl,gr=torch.func.grad(fn,argnums=(0,1))(l,r,dt)
    return local_contract(left,gl.detach().numpy()),local_contract(right,gr.detach().numpy())


def mask_sum(q,mask):return q[:,mask].sum(axis=1)


def normed(a,b,scale):return float(np.linalg.norm(a-b)/max(scale,1e-300))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--full-samples',type=int,default=17);p.add_argument('--fine-face-samples',type=int,default=257);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);rest=np.load(a.rest_cache)['rest'];state=np.load(a.state)
    dt=float(state['dt']);n=int(round(float(state['period_updates'])));T=n*dt;axis=int(state['axis']);sign=int(state['sign'])
    nodes=torch.tensor(state['nodes'],dtype=torch.float64);phase=torch.tensor(float(state['phase_per_step']),dtype=nodes.dtype)
    ref=torch.tensor(rest,dtype=nodes.dtype);theta=torch.tensor(float(np.arccos(1-dt**2*rank.R**2/2)),dtype=nodes.dtype)
    core=np.linalg.norm(unpack(rest)[0],axis=-1)>1e-12;receiver=np.roll(core,sign,axis)&~core;sender=core&~np.roll(core,sign,axis)
    def shape_layers(t):return tuple(x.detach().numpy() for x in native_layers(nodes,torch.as_tensor(t,dtype=nodes.dtype),n,phase,axis,sign))
    def face_fn(l,r,h):return action.SCALE*action.face_frame_oscillator_interval(l,r,h)
    checks=[]
    lrest=psi_clock(ref[None,:],-theta)[0].numpy();rrest=psi_clock(ref[None,:],theta)[0].numpy()
    qrest,_=face_charge(lrest,rest,dt);scale=max(float(np.linalg.norm(qrest[1])),1.)
    lr,cr,fr=shape_layers([.4]);rng=np.random.default_rng(806);rnd_l=lrest+.03*rng.normal(size=rest.size);rnd_r=rest+.03*rng.normal(size=rest.size)
    for label,l,r in [('rest',lrest,rest),('moving_seed',cr[0],fr[0]),('independent_perturbation',rnd_l,rnd_r)]:
        (ql,qr,flux,middle),_=face_charge(l,r,dt);adl,adr=ad_charge(face_fn,l,r,dt)
        checks.append({'case':label,'endpoint_momentum_relative_defect':max(normed(ql,adl,scale),normed(qr,adr,scale)),
                       'flux_divergence_relative_defect':normed(divergence(flux),ql+qr,scale),
                       'midpoint_flux_relative_defect':normed(flux,middle,scale),'global_flux_sum':np.sum(divergence(flux),axis=(1,2,3)).tolist()})
    full_times=np.linspace(0,1,a.full_samples);before,current,after=shape_layers(full_times);rows=[]
    for i,t in enumerate(full_times):
        ql0,qr0=ad_charge(action.interval,before[i],current[i],dt);ql1,qr1=ad_charge(action.interval,current[i],after[i],dt)
        qdelta=qr1-qr0;torque=ql1+qr1;residual=qdelta-torque
        face_prev,_=face_charge(before[i],current[i],dt);face_next,parts=face_charge(current[i],after[i],dt)
        df=face_next[1]-face_prev[1];tf=divergence(face_next[2]);rf=df-tf
        del_res=action.del_residual(torch.tensor(before[i]),torch.tensor(current[i]),torch.tensor(after[i]),dt).detach().numpy()
        exact=-local_contract(current[i],del_res)
        row={'fraction':float(t),'global_charge_previous':np.sum(qr0,axis=(1,2,3)).tolist(),'global_charge_next':np.sum(qr1,axis=(1,2,3)).tolist(),
             'full_local_balance_relative_defect':normed(residual,exact,scale),'full_global_torque':np.sum(torque,axis=(1,2,3)).tolist(),
             'receiver_charge_previous':mask_sum(qr0,receiver).tolist(),'receiver_charge_next':mask_sum(qr1,receiver).tolist(),
             'receiver_required_rate':(mask_sum(qdelta,receiver)/dt).tolist(),'receiver_actual_native_inflow_rate':(mask_sum(torque,receiver)/dt).tolist(),
             'receiver_face_inflow_rate':(mask_sum(tf,receiver)/dt).tolist(),'receiver_other_action_inflow_rate':(mask_sum(torque-tf,receiver)/dt).tolist(),
             'receiver_face_channel_inflow_rates':{k:float(np.sum(divergence(v[2][None])[0][receiver])/dt) for k,v in parts.items()},
             'sender_required_rate':(mask_sum(qdelta,sender)/dt).tolist(),'sender_actual_inflow_rate':(mask_sum(torque,sender)/dt).tolist(),
             'local_residual_norm_per_dt':(np.linalg.norm(residual.reshape(2,-1),axis=1)/dt).tolist(),
             'face_residual_norm_per_dt':(np.linalg.norm(rf.reshape(2,-1),axis=1)/dt).tolist()}
        rows.append(row);print(t,row['receiver_required_rate'],row['receiver_actual_native_inflow_rate'],flush=True)
    def integral(samples,field):return np.trapezoid(np.asarray([x[field] for x in samples]),x=full_times*T,axis=0)
    required=np.asarray(rows[-1]['receiver_charge_previous'])-rows[0]['receiver_charge_previous']
    total=integral(rows,'receiver_actual_native_inflow_rate');parts_total={}
    fine_rows=[]
    for count in (2*(a.full_samples-1)+1,a.fine_face_samples):
        tt=np.linspace(0,1,count);b,c,f=shape_layers(tt);values=[];channels=[]
        for j in range(count):
            next_,parts=face_charge(c[j],f[j],dt);values.append(mask_sum(divergence(next_[2]),receiver)/dt)
            channels.append([np.sum(divergence(parts[k][2][None])[0][receiver])/dt for k in ('Psi','W','chi')])
        ftotal=np.trapezoid(values,x=tt*T,axis=0);ch_total=np.trapezoid(channels,x=tt*T,axis=0)
        fine_rows.append({'samples':count,'receiver_face_integrated_inflow':ftotal.tolist(),'face_channels_integrated_inflow':dict(zip(('Psi','W','chi'),ch_total.tolist()))})
    face_at_end,_=face_charge(before[-1],current[-1],dt);face_at_start,_=face_charge(before[0],current[0],dt)
    result={'status':'V4_006_EXACT_NATIVE_FACE_PHASE_MOMENTUM_AND_HANDOFF_FLUX_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),
            'state_sha256':sha256(a.state),'rest_cache_sha256':sha256(a.rest_cache),'dt':dt,'period_updates':n,'axis':axis,'sign':sign,'face_formula_controls':checks,
            'receiver_site_count':int(receiver.sum()),'required_native_receiver_charge_gain':required.tolist(),'integrated_native_receiver_inflow':total.tolist(),
            'receiver_charge_gain_fraction_delivered':(total/np.where(abs(required)>1e-20,required,np.nan)).tolist(),
            'required_face_receiver_charge_gain':(mask_sum(face_at_end[1]-face_at_start[1],receiver)).tolist(),
            'full_integral_samples':len(full_times),'face_integral_refinement':fine_rows,'rows':rows,
            'claim_boundary':'Exact local phase identities and quadrature of the chosen candidate path. It is not a native orbit unless local balance and full DEL vanish; no fit or runtime correction.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
