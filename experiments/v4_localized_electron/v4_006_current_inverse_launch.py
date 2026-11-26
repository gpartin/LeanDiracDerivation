"""Inverse native phase-current launch at a selected handoff midpoint.

The full action gives charges and signed receiver influx. Solve both
charges and both receiver currents as an initial-data inverse problem;
then solve the actual backward DEL for the earlier layer. Subsequently
evolve the unmodified action, without resetting charges, frames or phase.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.linalg import lu_factor
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack,ad_charge,mask_sum
from v4_006_phase_corrected_handoff import native_layers,field_metrics,charge
from v4_006_current_action_relative_orbit import psi_clock
from v4_006_oriented_phase_exchange import phase_step
from v4_002_discrete_energy_gate import interval_energy


def main():
    p=argparse.ArgumentParser()
    for name in ("rest-cache","tangent-cache","state","output","state-output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fraction",type=float,default=.5);p.add_argument("--iterations",type=int,default=8)
    p.add_argument("--updates",type=int,default=3);p.add_argument("--energy",action="store_true")
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    ref=torch.tensor(np.load(a.rest_cache)["rest"],dtype=torch.float64)
    st=np.load(a.state)
    dt=float(st["dt"]);n=int(round(float(st["period_updates"])));axis=int(st["axis"]);sign=int(st["sign"])
    phase=torch.tensor(float(st["phase_per_step"]),dtype=ref.dtype);nodes=torch.tensor(st["nodes"],dtype=ref.dtype)
    theta=torch.tensor(np.arccos(1-dt*dt*rank.R**2/2),dtype=ref.dtype)
    pp,cc=unpack(ref.numpy());core=np.linalg.norm(pp,axis=-1)>1e-12;receiver=np.roll(core,sign,axis)&~core
    vacuum=cc[0,0,0];scale=np.sqrt(np.sum(abs(pp)**2)+np.sum(abs(cc-vacuum)**2))
    c=np.load(a.tangent_cache);F=lu_factor(c["following"]);P=lu_factor(c["previous"])
    prev,now,after=(v[0].detach() for v in native_layers(nodes,torch.tensor([a.fraction],dtype=ref.dtype),n,phase,axis,sign))
    qr=charge(psi_clock(ref[None],-theta)[0],ref,dt).detach()
    er=interval_energy(psi_clock(ref[None],-theta)[0].numpy(),ref.numpy(),dt,2.5e-5)
    _,q0=ad_charge(action.interval,prev.numpy(),now.numpy(),dt);ql,q1=ad_charge(action.interval,now.numpy(),after.numpy(),dt)
    target_flux=torch.tensor(mask_sum(q1-q0,receiver),dtype=ref.dtype)
    target=torch.cat((qr,target_flux))
    # Current/charge are both derivatives with respect to native phases.
    mask=torch.tensor(receiver,dtype=ref.dtype)
    def gen(raw,family,mask=None):
        q=raw.reshape(5,5,5,4,3);z=torch.zeros_like(q[...,0,:])
        if family==0: g=torch.stack((-q[...,1,:],q[...,0,:],z,z),dim=-2)
        else:g=torch.stack((z,z,-q[...,3,:],q[...,2,:]),dim=-2)
        if mask is not None:g=g*mask[...,None,None]
        return g.reshape(-1)
    def metrics(r):
        ml,mr=torch.func.grad(action.interval,argnums=(0,1))(now,r,dt)
        charges=torch.stack([torch.dot(mr,gen(r,k)) for k in (0,1)])
        flux=torch.stack([torch.dot(ml,gen(now,k,mask))+torch.dot(mr,gen(r,k,mask)) for k in (0,1)])
        result=torch.cat((charges,flux))
        if a.energy:
            h=dt*2.5e-5
            e=-(action.interval(now,r,dt+h)-action.interval(now,r,dt-h))/(2*h)
            result=torch.cat((result,e[None]))
        return result
    if a.energy:target=torch.cat((target,target.new_tensor([er])))
    normal=torch.clamp(abs(target),min=1.)
    def residual(r):return (metrics(r)-target)/normal
    r=after.clone();trace=[]
    for i in range(a.iterations):
        f=residual(r).detach();e=float(torch.linalg.vector_norm(f));trace.append(e)
        print("inverse",i,e,flush=True)
        if e<1e-9:break
        j=torch.func.jacrev(residual)(r).detach().numpy()
        # Minimum Euclidean-norm local perturbation; no runtime constraint.
        v=np.linalg.lstsq(j,-f.numpy(),rcond=1e-11)[0]
        for b in range(10):
            cand=r+torch.tensor(v*2.**(-b),dtype=r.dtype)
            if float(torch.linalg.vector_norm(residual(cand)))<e:r=cand;break
        else:break
    def info(l,m):
        left,right=ad_charge(action.interval,l.numpy(),m.numpy(),dt)
        return {"global_charge_relative_defect":((right.sum(axis=(1,2,3))-qr.numpy())/abs(qr.numpy())).tolist(),
            "receiver_inflow_rate":(mask_sum(left+right,receiver)/dt).tolist(),
            "interval_energy_relative_to_rest":interval_energy(l.numpy(),m.numpy(),dt,2.5e-5)/er}
    correction=float(torch.linalg.vector_norm(r-after))/scale
    correction_relative_envelope=float(torch.linalg.vector_norm(r-after))/max(float(torch.linalg.vector_norm(after-psi_clock(now[None],theta)[0])),1e-30)
    reverse,backtrace,backerr=phase_step(r,now,prev,dt,P,scale,6,backward=True)
    layers=[reverse,now,r];report=[dict(step=0,**info(now,r),exact_full_DEL_defect=backerr)]
    if backerr<1e-6 and float(torch.linalg.vector_norm(residual(r)))<1e-7:
        first=now;current=r
        for step in range(1,a.updates+1):
            guess=2*current-first
            nxt,tr,err=phase_step(first,current,guess,dt,F,scale,5)
            d=info(current,nxt);d.update(step=step,exact_full_DEL_defect=err,newton_trace=tr,
                        flux_fraction_of_target=(np.asarray(d["receiver_inflow_rate"])/(target_flux.numpy()/dt)).tolist())
            print("free",step,"DEL",err,"flux ratio",d["flux_fraction_of_target"],flush=True)
            report.append(d);layers.append(nxt)
            if err>1e-6:break
            first,current=current,nxt
    out={"status":"V4_006_TWO_CHARGE_CURRENT_INVERSE_LAUNCH_SCREEN",
         "action_sha256":sha256(Path(action.__file__)),"state_sha256":sha256(a.state),"rest_cache_sha256":sha256(a.rest_cache),
         "tangent_cache_sha256":sha256(a.tangent_cache),"dt":dt,"fraction":a.fraction,"axis":axis,"sign":sign,"period_updates":n,
         "energy_constraint":a.energy,"initial_inverse_residual":trace[0],"final_inverse_residual":float(torch.linalg.vector_norm(residual(r))),
         "inverse_trace":trace,"target_receiver_rate":(target_flux/dt).tolist(),
         "old_receiver_rate":(mask_sum(ql+q1,receiver)/dt).tolist(),
         "following_layer_correction_relative_to_rest_contrast":correction,"correction_relative_to_original_slow_motion":correction_relative_envelope,
         "backward_DEL_defect":backerr,"backward_trace":backtrace,
         "native_initial_pair":info(reverse,now),"native_steps":report,
         "claim_boundary":"Inverse prepared midpoint current from native action. Only initial data are changed. Three unrestricted updates do not certify a complete crossing."}
    np.savez_compressed(a.state_output,states=np.stack([x.numpy() for x in layers]),dt=dt)
    out["output_state_sha256"]=sha256(a.state_output)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
