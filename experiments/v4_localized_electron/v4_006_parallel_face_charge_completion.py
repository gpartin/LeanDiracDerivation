"""Screen equal-weight completion of the oriented native frame's phase channels.

W=chi wedge Psi carries the sum phase, S=chi†Psi carries the difference.
The latter is already in the repository; test its face oscillator with the
same native B as W, not a fitted scale. No action is installed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_phase_corrected_handoff import native_layers
from v4_006_face_phase_momentum import ad_charge, local_contract, unpack, mask_sum, one_field, divergence
from v4_006_current_action_relative_orbit import psi_clock


def main():
    ap=argparse.ArgumentParser()
    for name in ("rest-cache","tangent-cache","state","output"):ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--samples",type=int,default=33)
    a=ap.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    ref=torch.tensor(np.load(a.rest_cache)["rest"],dtype=torch.float64)
    st=np.load(a.state);dt=float(st["dt"]);n=int(round(float(st["period_updates"])))
    axis=int(st["axis"]);sign=int(st["sign"]);phase=torch.tensor(float(st["phase_per_step"]),dtype=ref.dtype);nodes=torch.tensor(st["nodes"],dtype=ref.dtype)
    theta=torch.tensor(np.arccos(1-dt*dt*rank.R**2/2),dtype=ref.dtype)
    p,c=unpack(ref.numpy());vac=c[0,0,0];core=np.linalg.norm(p,axis=-1)>1e-12;receiver=np.roll(core,sign,axis)&~core
    scale=np.sqrt(np.sum(abs(p)**2)+np.sum(abs(c-vac)**2))
    factor=lu_factor(np.load(a.tangent_cache)["following"])
    def complement(l,r,h):return rank.B*action.parallel_overlap_face_interval(l,r,h)
    def candidate(l,r,h):return action.interval(l,r,h)+complement(l,r,h)
    def force(fn,l,m,r):
        def summed(x,y,z):return fn(x,y,dt)+fn(y,z,dt)
        return torch.func.grad(summed,argnums=1)(l,m,r).detach().numpy()
    def rest_def(fn,l,m,r):return float(np.linalg.norm(lu_solve(factor,force(fn,l,m,r)))/dt**2/scale)
    lrest=psi_clock(ref[None],-theta)[0];rrest=psi_clock(ref[None],theta)[0]
    qrest=ad_charge(action.interval,lrest.numpy(),ref.numpy(),dt)[1]
    qscale=np.linalg.norm(qrest)
    def sfield(x):
        p,c=unpack(x);return np.sum(c.conj()*p,axis=-1,keepdims=True)/rank.R
    rng=np.random.default_rng(62)
    left=(lrest.numpy()+.03*rng.normal(size=ref.numel()));right=ref.numpy()+.03*rng.normal(size=ref.numel())
    ss=one_field(sfield(left),sfield(right),dt,rank.B,1-dt*dt*rank.R**2/2)
    al,ar=ad_charge(complement,left,right,dt)
    identity=max(np.linalg.norm(al-np.stack((ss[0],-ss[0]))),np.linalg.norm(ar-np.stack((ss[1],-ss[1]))))/qscale
    times=torch.linspace(0,1,a.samples,dtype=ref.dtype);b,m,f=native_layers(nodes,times,n,phase,axis,sign);rows=[]
    for t,l,q,r in zip(times,b,m,f):
        def term(fn):
            _,qr0=ad_charge(fn,l.numpy(),q.numpy(),dt)
            ql1,qr1=ad_charge(fn,q.numpy(),r.numpy(),dt)
            return mask_sum(qr1-qr0,receiver)/dt,mask_sum(ql1+qr1,receiver)/dt
        bs,bi=term(action.interval);cs,ci=term(complement)
        bf=force(action.interval,l,q,r);cf=force(complement,l,q,r)
        rows.append({"fraction":float(t),"old_storage_rate":bs.tolist(),"old_inflow_rate":bi.tolist(),
                     "parallel_storage_rate":cs.tolist(),"parallel_inflow_rate":ci.tolist(),
                     "old_DEL_defect":float(np.linalg.norm(lu_solve(factor,bf))/dt**2/scale),
                     "candidate_DEL_defect":float(np.linalg.norm(lu_solve(factor,bf+cf))/dt**2/scale)})
    def integ(key):return np.trapezoid(np.array([r[key] for r in rows]),x=times.numpy()*n*dt,axis=0)
    bq,bi,cq,ci=(integ(k) for k in ("old_storage_rate","old_inflow_rate","parallel_storage_rate","parallel_inflow_rate"))
    out={"status":"V4_006_EQUAL_NATIVE_PARALLEL_FACE_COMPLETION_NOT_ADMITTED",
         "action_sha256":sha256(Path(action.__file__)),"state_sha256":sha256(a.state),"rest_cache_sha256":sha256(a.rest_cache),
         "tangent_cache_sha256":sha256(a.tangent_cache),"scale":rank.B,"dt":dt,"period_updates":n,"axis":axis,"sign":sign,
         "opposite_charge_formula_relative_defect":float(identity),
         "rest_baseline_DEL_defect":rest_def(action.interval,lrest,ref,rrest),
         "rest_candidate_DEL_defect":rest_def(candidate,lrest,ref,rrest),
         "old_required":bq.tolist(),"old_inflow":bi.tolist(),"parallel_additional_required":cq.tolist(),"parallel_additional_inflow":ci.tolist(),
         "old_fraction_delivered":(bi/bq).tolist(),"candidate_fraction_delivered":((bi+ci)/(bq+cq)).tolist(),
         "old_maximum_path_defect":max(r["old_DEL_defect"] for r in rows),"candidate_maximum_path_defect":max(r["candidate_DEL_defect"] for r in rows),
         "rows":rows,"claim_boundary":"Geometric equal-weight face completion: opposite phase torque derived from chi†Psi. Evaluated on the same prior collocation path. Not an admitted action or evolved carrier."}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
