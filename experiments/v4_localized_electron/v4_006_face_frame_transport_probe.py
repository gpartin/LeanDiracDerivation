"""Test the derived conservative face-frame replacement before admission."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
import v4_006_face_frame_transport_action as candidate
from v4_003_fixed_phase_space_readout import sha256
from v4_006_current_action_relative_orbit import psi_clock
from v4_006_phase_corrected_handoff import native_layers
from v4_006_face_phase_momentum import ad_charge,unpack,mask_sum


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--baseline-flux',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank.lock_inertia();torch.set_num_threads(2);rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64);state=np.load(a.state)
    nodes=torch.tensor(state['nodes'],dtype=ref.dtype);dt=float(state['dt']);n=int(round(float(state['period_updates'])));axis=int(state['axis']);sign=int(state['sign'])
    phase=torch.tensor(float(state['phase_per_step']),dtype=ref.dtype);theta=torch.tensor(float(np.arccos(1-(dt*rank.R)**2/2)),dtype=ref.dtype)
    core=np.linalg.norm(unpack(rest)[0],axis=-1)>1e-12;receiver=np.roll(core,sign,axis)&~core
    source=json.loads(a.baseline_flux.read_text());target=np.asarray(source['required_native_receiver_charge_gain']);flow=np.asarray(source['integrated_native_receiver_inflow'])
    ts=np.asarray([r['fraction'] for r in source['rows']]);before,current,after=native_layers(nodes,torch.tensor(ts,dtype=ref.dtype),n,phase,axis,sign)
    factor=lu_factor(np.load(a.tangent_cache)['following']);vac=rest.reshape(125,12)[0].copy();vac[:6]=0;scale=np.linalg.norm(rest-np.tile(vac,125));rows=[]
    l0=psi_clock(ref[None,:],-theta)[0];r0=psi_clock(ref[None,:],theta)[0];base_rest=action.del_residual(l0,ref,r0,dt)
    base_data=[action.del_residual(before[i],current[i],after[i],dt).detach().numpy() for i in range(len(ts))]
    rng=np.random.default_rng(621);random=torch.tensor(rng.normal(size=(5,5,5,3))+1j*rng.normal(size=(5,5,5,3)),dtype=torch.complex128);vel=torch.tensor(rng.normal(size=(5,5,5,3)),dtype=ref.dtype)
    for mode in ('plain','aligned'):
        def inc(l,r,h):return candidate.delta_interval(l,r,h,mode)
        def del_delta(l,q,r):return torch.func.grad(lambda x,y,z:inc(x,y,dt)+inc(y,z,dt),argnums=1)(l,q,r)
        dd=del_delta(l0,ref,r0);rest_err=float(np.linalg.norm(lu_solve(factor,(base_rest+dd).detach().numpy()))/dt**2/scale)
        ad=candidate.skew_face_advection(random,vel,mode=='aligned');test=torch.sum(random.conj()*ad,dim=-1)
        moments=[];rate=[];defects=[]
        for i,frac in enumerate(ts):
            lp,rp=ad_charge(inc,before[i].numpy(),current[i].numpy(),dt);lf,rf=ad_charge(inc,current[i].numpy(),after[i].numpy(),dt)
            moments.append(mask_sum(rp,receiver));rate.append(mask_sum(lf+rf,receiver)/dt)
            base=base_data[i];delta=del_delta(before[i],current[i],after[i]).detach().numpy();defects.append([float(np.linalg.norm(lu_solve(factor,x))/dt**2/scale) for x in (base,base+delta)])
        dtarget=np.asarray(moments[-1])-moments[0];dflow=np.trapezoid(rate,x=ts*n*dt,axis=0);de=np.asarray(defects)
        row={'mode':mode,'generic_real_inner_product_defect':float(abs(test.real.sum())/torch.linalg.vector_norm(random)/torch.linalg.vector_norm(ad)),
             'generic_local_phase_cross_defect':float(torch.linalg.vector_norm(test.imag)/(torch.linalg.vector_norm(random)*torch.linalg.vector_norm(ad))),
             'rest_delta_DEL_norm':float(torch.linalg.vector_norm(dd)),'rest_candidate_preconditioned_acceleration_defect':rest_err,
             'required_candidate_receiver_gain':(target+dtarget).tolist(),'integrated_candidate_receiver_inflow':(flow+dflow).tolist(),
             'baseline_fraction_delivered':(flow/target).tolist(),'candidate_fraction_delivered':((flow+dflow)/(target+dtarget)).tolist(),
             'additional_candidate_inflow':dflow.tolist(),'maximum_baseline_path_defect':float(np.max(de[:,0])),'maximum_candidate_path_defect':float(np.max(de[:,1]))}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'status':'V4_006_CONSERVATIVE_FACE_FRAME_ACTION_REPLACEMENT_SCREEN_NOT_ADMITTED','action_sha256':sha256(Path(action.__file__)),
            'candidate_sha256':sha256(Path(candidate.__file__)),'state_sha256':sha256(a.state),'tangent_cache_sha256':sha256(a.tangent_cache),'dt':dt,'rows':rows,
            'claim_boundary':'Native local frame transport comparison, scope limited to same-state phase momentum and rest/dynamic residual. Not installed, not a stage pass.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
