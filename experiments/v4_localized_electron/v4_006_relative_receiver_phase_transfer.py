"""Test the native phase-momentum sign by varying the relative endpoint phase.

The exact face inflow at leading order is K Re(F_j† u_i - u_j† F_i).
For a donor losing amplitude and a receiver gaining amplitude, aligned
fields can route this phase momentum backwards; opposite-phase fields
reverse its sign. The translated endpoint already permits a freely solved
common Psi phase. Test that boundary-phase degree of freedom, not a new
action or runtime phase rule. Chi's ordinary exterior remains unchanged.
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
from v4_006_charge_reduced_handoff import rotation
from v4_006_face_phase_momentum import face_charge,divergence,unpack,mask_sum
from v4_006_phase_corrected_handoff import project_face_capacity


def path(ref,t,axis,sign,phase):
    # Floquet chart: after one cell, translate both fields and rotate Psi
    # by its free relative endpoint phase. This is only a trial shape path.
    t=np.asarray(t);k=np.floor(t).astype(int);fraction=t-k
    translated=torch.roll(ref.reshape(5,5,5,12),sign,axis).reshape(-1)
    translated=rotation(translated,torch.tensor([phase,0.],dtype=ref.dtype))
    f=torch.tensor((1-np.cos(np.pi*fraction))/2,dtype=ref.dtype)
    samples=(1-f[:,None])*ref+f[:,None]*translated
    samples=project_face_capacity(samples,ref)
    result=[]
    for i,m in enumerate(k):
        temp=torch.roll(samples[i].reshape(5,5,5,12),sign*int(m),axis).reshape(-1)
        result.append(rotation(temp,torch.tensor([phase*int(m),0.],dtype=ref.dtype)))
    return torch.stack(result).detach().numpy()


def evaluate(ref,axis,sign,phi,n,dt,count):
    theta=float(np.arccos(1-(dt*rank.R)**2/2));t=np.linspace(0,1,count);h=1/n
    receiver_core=np.linalg.norm(unpack(ref.numpy())[0],axis=-1)>1e-12;receiver=np.roll(receiver_core,sign,axis)&~receiver_core
    try:b,q,f=(path(ref,t+z*h,axis,sign,phi) for z in (-1,0,1))
    except ValueError as e:return {'axis':axis,'sign':sign,'endpoint_Psi_phase':phi,'samples':count,'valid_face_capacity_chart':False,'reason':str(e)}
    p=np.asarray([rotation(torch.tensor(v),torch.tensor([-theta,0.],dtype=ref.dtype)).numpy() for v in b])
    r=np.asarray([rotation(torch.tensor(v),torch.tensor([theta,0.],dtype=ref.dtype)).numpy() for v in f])
    values=[];required=[];channels=[];axis_flows=[]
    for i in range(count):
        old,_=face_charge(p[i],q[i],dt);new,parts=face_charge(q[i],r[i],dt)
        values.append(mask_sum(divergence(new[2]),receiver)/dt)
        axis_flows.append(np.stack([mask_sum(np.roll(new[2][:,k],1,k+1)-new[2][:,k],receiver)/dt for k in range(3)]))
        required.append(mask_sum(old[1],receiver))
        channels.append([float(np.sum(divergence(parts[k][2][None])[0][receiver])/dt) for k in ('Psi','W','chi')])
    target=np.asarray(required[-1])-required[0];flow=np.trapezoid(values,x=t*n*dt,axis=0);channel=np.trapezoid(channels,x=t*n*dt,axis=0)
    axial_integrals=np.trapezoid(axis_flows,x=t*n*dt,axis=0)
    rp,rc=unpack(ref.numpy());rw=np.cross(rc,rp)/rank.R;K=rank.B*np.sin(theta)/dt
    required_by_axis=[]
    for k in range(3):
        def c(f):return K*np.real(np.sum(f.conj()*(2*f-np.roll(f,-1,k)-np.roll(f,1,k)),axis=-1))
        old=np.stack((c(rp)+c(rw),c(rw)));new=np.roll(old,sign,axis+1)
        required_by_axis.append(mask_sum(new-old,receiver))
    required_by_axis=np.asarray(required_by_axis)
    return {'axis':axis,'sign':sign,'endpoint_Psi_phase':phi,'samples':count,'valid_face_capacity_chart':True,
            'required_face_gain':target.tolist(),'integrated_face_inflow':flow.tolist(),'fraction_delivered':(flow/target).tolist(),
            'channel_integrated_inflow':dict(zip(('Psi','W','chi'),channel.tolist())),
            'receiver_integrated_inflow_by_bond_axis':axial_integrals.tolist(),
            'receiver_static_endpoint_charge_gain_by_bond_axis':required_by_axis.tolist(),
            'fraction_static_endpoint_charge_stored_in_motion_axis':(required_by_axis[axis]/required_by_axis.sum(axis=0)).tolist(),
            'normalized_two_charge_balance_defect':float(np.linalg.norm((flow-target)/target))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--dt',type=float,default=1/256)
    p.add_argument('--period-updates',type=int,default=17564);p.add_argument('--samples',type=int,default=65)
    p.add_argument('--phases',type=float,nargs='+',default=[0,np.pi/2,np.pi,-np.pi/2]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-output',type=Path,help='save the derived antiphase initial chart on x-negative for the native BVP')
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=torch.float64);rows=[]
    for axis in range(3):
        for sign in (-1,1):
            for phase in a.phases:
                row=evaluate(ref,axis,sign,phase,a.period_updates,a.dt,a.samples);rows.append(row)
                print(axis,sign,phase,row.get('fraction_delivered',row.get('reason')),flush=True)
    refined=[]
    for axis in range(3):
        for sign in (-1,1):
            choices=[r for r in rows if r['axis']==axis and r['sign']==sign and r['valid_face_capacity_chart']]
            if choices:
                best=min(choices,key=lambda r:r['normalized_two_charge_balance_defect']);refined.append(evaluate(ref,axis,sign,best['endpoint_Psi_phase'],a.period_updates,a.dt,4*(a.samples-1)+1))
    result={'status':'V4_006_RELATIVE_RECEIVER_PHASE_NATIVE_FLUX_SCREEN','action_sha256':sha256(Path(action.__file__)),'rest_cache_sha256':sha256(a.rest_cache),
            'dt':a.dt,'period_updates':a.period_updates,'rows':rows,'refined_best_by_axis_sign':refined,
            'claim_boundary':'Boundary phase is a free preparation parameter, not an action coefficient. This compares exact native face flux on charge-capacity trial paths; not a solved DEL orbit or total-current test.'}
    if a.seed_output:
        from v4_006_current_action_relative_orbit import shift
        t=np.arange(5)/5;phi=np.pi;full=torch.tensor(path(ref,t,0,-1,phi),dtype=ref.dtype)
        # q(t)=R_{theta*n*t+phi*t} tau(t) periodic(t), with phi
        # a boundary phase selected by native charge-flow sign.
        unwound=torch.stack([rotation(full[i],torch.tensor([-phi*x,0.],dtype=ref.dtype)) for i,x in enumerate(t)])
        nodes=shift(unwound,torch.tensor(t,dtype=ref.dtype),0).numpy()
        theta=float(np.arccos(1-(a.dt*rank.R)**2/2))
        np.savez_compressed(a.seed_output,nodes=nodes,period_updates=a.period_updates,phase_per_step=theta+phi/a.period_updates,dt=a.dt,axis=0,sign=-1)
        result['antiphase_BVP_seed_sha256']=sha256(a.seed_output)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
