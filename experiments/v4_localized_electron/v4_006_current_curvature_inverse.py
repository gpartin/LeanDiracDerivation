"""Match native current and two successive free advances at matched energy.

Only the first committed layer is varied. Both later layers satisfy the
actual implicit Euler--Lagrange equations. The adjoint differentiates
through both native roots, including the dependence on the first layer
through both paths. No target is used by the time update.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json
import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse.linalg import LinearOperator,gmres
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack,ad_charge,mask_sum
from v4_006_phase_corrected_handoff import native_layers,reduced_del,reduced_frozen_interval,charge
from v4_006_current_action_relative_orbit import psi_clock
from v4_006_oriented_phase_exchange import phase_step
from v4_002_discrete_energy_gate import interval_energy


def main():
    p=argparse.ArgumentParser()
    for name in ("rest-cache","tangent-cache","state","warm-start","output","state-output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fraction",type=float,default=.5)
    p.add_argument("--iterations",type=int,default=7)
    p.add_argument("--updates",type=int,default=5)
    p.add_argument("--energy-ratio",type=float,help="matched native diagnostic energy relative to resting reference; not a Dirac energy law")
    a=p.parse_args()
    rank.lock_inertia();torch.set_num_threads(2)
    rest=torch.tensor(np.load(a.rest_cache)["rest"],dtype=torch.float64);st=np.load(a.state)
    dt=float(st["dt"]);n=int(round(float(st["period_updates"])));axis=int(st["axis"]);sign=int(st["sign"])
    phase=torch.tensor(float(st["phase_per_step"]),dtype=rest.dtype)
    nodes=torch.tensor(st["nodes"],dtype=rest.dtype)
    theta=torch.tensor(np.arccos(1-dt*dt*rank.R**2/2),dtype=rest.dtype)
    rp,rc=unpack(rest.numpy());core=np.linalg.norm(rp,axis=-1)>1e-12
    receiver=np.roll(core,sign,axis)&~core
    vacuum=rc[0,0,0];scale=np.sqrt(np.sum(abs(rp)**2)+np.sum(abs(rc-vacuum)**2))
    c=np.load(a.tangent_cache);F=lu_factor(c["following"]);P=lu_factor(c["previous"])
    refcharge=charge(psi_clock(rest[None],-theta)[0],rest,dt).detach()
    er=interval_energy(psi_clock(rest[None],-theta)[0].numpy(),rest.numpy(),dt,2.5e-5)
    chart=native_layers(nodes,torch.tensor([a.fraction,a.fraction+1/n,a.fraction+2/n],dtype=rest.dtype),n,phase,axis,sign)
    expected=[]
    for l,m,f in zip(*chart):
        _,qold=ad_charge(action.interval,l.numpy(),m.numpy(),dt)
        _,qnew=ad_charge(action.interval,m.numpy(),f.numpy(),dt)
        expected.append(mask_sum(qnew-qold,receiver))
    expected=np.array(expected)
    warm=np.load(a.warm_start)["states"]
    previous,q,r,s,t=(torch.tensor(x,dtype=rest.dtype) for x in warm[:5])
    if float(torch.linalg.vector_norm(q-chart[1][0]))>1e-10:
        raise ValueError("warm-start midpoint does not match the reference chart")
    r0=r.clone()
    mask=torch.tensor(receiver,dtype=rest.dtype)
    def gen(raw,family,local=False):
        v=raw.reshape(5,5,5,4,3);z=torch.zeros_like(v[...,0,:])
        if family==0:g=torch.stack((-v[...,1,:],v[...,0,:],z,z),dim=-2)
        else:g=torch.stack((z,z,-v[...,3,:],v[...,2,:]),dim=-2)
        if local:g=g*mask[...,None,None]
        return g.reshape(-1)
    def measures(l,r):
        ml,mr=torch.func.grad(reduced_frozen_interval,argnums=(0,1))(l,r,dt)
        ch=torch.stack([torch.dot(mr,gen(r,k)) for k in (0,1)])
        curr=torch.stack([torch.dot(ml,gen(l,k,True))+torch.dot(mr,gen(r,k,True)) for k in (0,1)])
        return ch,curr
    def energy(l,r):
        h=dt*2.5e-5
        return -(reduced_frozen_interval(l,r,dt+h)-reduced_frozen_interval(l,r,dt-h))/(2*h)
    e0=torch.tensor(expected[0],dtype=rest.dtype);e1=torch.tensor(expected[1],dtype=rest.dtype);e2=torch.tensor(expected[2],dtype=rest.dtype)
    norm=torch.clamp(abs(e0),min=1.)
    def residual(r,s,t):
        ch,j=measures(q,r);_,j1=measures(r,s);_,j2=measures(s,t)
        result=torch.cat(((ch-refcharge)/abs(refcharge),(j-e0)/norm,
                          ((j1-j)-(e1-e0))/norm,((j2-j1)-(e2-e1))/norm))
        if a.energy_ratio is not None:
            result=torch.cat((result,((energy(q,r)/er-a.energy_ratio)[None])))
        return result
    def native_root(l,r,guess):
        return phase_step(l,r,guess,dt,F,scale,6)
    def implicit_rows(r,s,t):
        deriv=torch.func.jacrev(residual,argnums=(0,1,2))(r,s,t)
        deriv=[x.detach().numpy() for x in deriv]
        _,pull0=torch.func.vjp(lambda x,y:reduced_del(q,x,y,dt),r,s)
        _,pull1=torch.func.vjp(lambda x,y,z:reduced_del(x,y,z,dt),r,s,t)
        def solve(pull,h):
            if np.linalg.norm(h)<1e-12:return None,0.,0
            def mat(v):
                x=lu_solve(F,v,trans=1)
                return pull(torch.tensor(x,dtype=rest.dtype))[-1].detach().numpy()
            op=LinearOperator((r.numel(),r.numel()),matvec=mat,dtype=np.float64)
            count=[0]
            y,info=gmres(op,h,rtol=2e-6,atol=0.,restart=25,maxiter=4,
                         callback=lambda _:count.__setitem__(0,count[0]+1),callback_type="pr_norm")
            defect=float(np.linalg.norm(mat(y)-h)/np.linalg.norm(h))
            if not np.isfinite(defect) or defect>2e-4:raise RuntimeError(f"adjoint solve failed: {info}, {defect}")
            x=lu_solve(F,y,trans=1)
            return [v.detach().numpy() for v in pull(torch.tensor(x,dtype=rest.dtype))],defect,count[0]
        rows=[];diag=[]
        for j in range(deriv[0].shape[0]):
            dr,ds,dt_= [v[j].copy() for v in deriv]
            correction,err1,it1=solve(pull1,dt_)
            if correction is not None:
                dr-=correction[0];ds-=correction[1]
            correction,err0,it0=solve(pull0,ds)
            if correction is not None:dr-=correction[0]
            rows.append(dr);diag.append({"row":j,"adjoint_relative_defect":max(err0,err1),"iterations":it0+it1})
        return np.array(rows),diag
    trace=[]
    # Native initial root is taken from the already verified free archive.
    start_del=max(float(np.linalg.norm(lu_solve(F,action.del_residual(l,m,h,dt).detach().numpy()))/scale/dt**2) for l,m,h in ((q,r,s),(r,s,t)))
    if start_del>1e-6:raise ValueError(f"warm start not native: {start_del}")
    for step in range(a.iterations):
        f=residual(r,s,t).detach().numpy();err=float(np.linalg.norm(f))
        row={"iteration":step,"residual":f.tolist(),"norm":err,"energy_relative_to_rest":float(energy(q,r)/er)}
        trace.append(row);print("shoot",step,"norm",err,"error",f.tolist(),"energy",row["energy_relative_to_rest"],flush=True)
        if err<1e-8:break
        J,adj=implicit_rows(r,s,t);row["adjoints"]=adj
        v,singular,_=np.linalg.svd(J,full_matrices=False)
        row["singular_values"]=singular.tolist()
        d=np.linalg.lstsq(J,-f,rcond=1e-11)[0]
        # Advancing initial-layer correction; the second-layer guess is
        # extrapolation only and is replaced by the actual DEL root.
        for bisect in range(8):
            delta=torch.tensor(d*2.**(-bisect),dtype=r.dtype)
            rr=r+delta;guess=s+2*delta
            ss,nt,ne=native_root(q,rr,guess)
            if ne>1e-6:continue
            tt,nt2,ne2=native_root(rr,ss,t+3*delta)
            ne=max(ne,ne2)
            value=float(torch.linalg.vector_norm(residual(rr,ss,tt)))
            if ne<1e-6 and value<err:
                if step==0 and bisect==0:
                    # Nonlinear matched perturbation checks the implicit row
                    # derivative, rather than assuming the rest tangent.
                    predicted=J@delta.numpy();observed=residual(rr,ss,tt).detach().numpy()-f
                    row["linear_prediction_defect"]=float(np.linalg.norm(observed-predicted)/max(np.linalg.norm(predicted),1e-300))
                r,s,t=rr,ss,tt;row.update(accepted_scale=2.**(-bisect),accepted_native_DEL_defect=ne,accepted_newton_trace=nt);break
            print("line",step,bisect,"native",ne,"metric",value,flush=True)
        else:
            row["line_search_failed"]=True;break
    final=float(torch.linalg.vector_norm(residual(r,s,t)))
    # No more control after initial pair is chosen. The actual previous layer
    # is solved backward rather than taken from the target path.
    reverse,backtrace,backerr=phase_step(r,q,previous,dt,P,scale,6,backward=True)
    def info(l,m):
        ql,qr=ad_charge(action.interval,l.numpy(),m.numpy(),dt)
        nativej=mask_sum(ql+qr,receiver)
        qdef=(qr.sum(axis=(1,2,3))-refcharge.numpy())/abs(refcharge.numpy())
        e=interval_energy(l.numpy(),m.numpy(),dt,2.5e-5)/er
        return {"global_charge_relative_defect":qdef.tolist(),"receiver_inflow_rate":(nativej/dt).tolist(),
                "interval_energy_relative_to_rest":e,"flux_fraction_of_midpoint_target":(nativej/expected[0]).tolist()}
    rows=[{"step":0,**info(q,r),"exact_full_DEL_defect":backerr}]
    layers=[reverse,q,r]
    # If optimizer stalled, retain the measured partial launch as a diagnostic
    # but do not falsely label a repaired initial pair. Full native solve
    # is the only acceptance condition for each step.
    if backerr<1e-6:
        left=q;middle=r
        for i in range(1,a.updates+1):
            guess=s if i==1 else (t if i==2 else 2*middle-left)
            next_,nt,ne=phase_step(left,middle,guess,dt,F,scale,5)
            d=info(middle,next_)
            d.update(step=i,exact_full_DEL_defect=ne,newton_trace=nt)
            if i<len(expected):d["flux_fraction_of_consecutive_chart_target"]=(np.array(d["receiver_inflow_rate"])*dt/expected[i]).tolist()
            rows.append(d);layers.append(next_)
            print("release",i,"DEL",ne,"current",d["flux_fraction_of_midpoint_target"],flush=True)
            if ne>1e-6:break
            left,middle=middle,next_
    out={"status":"V4_006_NATIVE_CURRENT_AND_CURVATURE_INITIAL_DATA_SCREEN",
         "action_sha256":sha256(Path(action.__file__)),"rest_cache_sha256":sha256(a.rest_cache),
         "tangent_cache_sha256":sha256(a.tangent_cache),"state_sha256":sha256(a.state),"warm_start_sha256":sha256(a.warm_start),
         "dt":dt,"axis":axis,"sign":sign,"fraction":a.fraction,"period_updates":n,
         "expected_chart_receiver_rate_by_step":(expected/dt).tolist(),
         "expected_first_rate_advance":((expected[1]-expected[0])/dt).tolist(),
         "energy_ratio_constraint":a.energy_ratio,
         "rest_interval_energy":er,
         "warm_start_interval_energy_relative_to_rest":interval_energy(q.numpy(),r0.numpy(),dt,2.5e-5)/er,
         "initial_native_DEL_defect":start_del,"trace":trace,"final_residual":residual(r,s,t).detach().tolist(),
         "final_constraint_norm":final,"backward_DEL_defect":backerr,"backward_trace":backtrace,
         "initial_pair":info(reverse,q),"initial_layer_change_relative_rest_contrast":float(torch.linalg.vector_norm(r-r0)/scale),
         "native_steps":rows,"claim_boundary":"Both currents and two free advances are an inverse initial-data screen at one midpoint. Targets are measured on the prior chart at the independent comparator crossing interval, not a proven Dirac-current map of the translating nonlinear carrier. No action change or runtime constraint; this is not a full-cell orbit."}
    np.savez_compressed(a.state_output,states=np.stack([x.numpy() for x in layers]),dt=dt)
    out["output_state_sha256"]=sha256(a.state_output)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__":main()
