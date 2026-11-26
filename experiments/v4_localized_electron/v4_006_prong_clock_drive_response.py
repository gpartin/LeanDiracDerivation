"""Native direct-prong coefficient sensitivity of the sustained clock current.

The positive direct-prong kinetic metric and the matched face oscillator
presently share B. Vary their existing relative coefficient as a diagnostic,
not a calibrated action. Differentiate the actual implicit native steps,
then optionally test a finite reciprocal ablation on the same initial pair.
"""
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
from v4_006_phase_corrected_handoff import reduced_del,reduced_frozen_interval
from v4_006_current_action_relative_orbit import psi_clock

def main():
    p=argparse.ArgumentParser()
    for name in ("rest-cache","tangent-cache","archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--scales",type=float,nargs="*",default=[])
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    ref=torch.tensor(np.load(a.rest_cache)["rest"],dtype=torch.float64)
    archive=np.load(a.archive);dt=float(archive["dt"]);states=[torch.tensor(x,dtype=ref.dtype) for x in archive["states"]]
    q,r,s,t=states[1:5];pp,cc=unpack(ref.numpy());core=np.linalg.norm(pp,axis=-1)>1e-12
    receiver=np.roll(core,-1,axis=0)&~core;mask=torch.tensor(receiver,dtype=ref.dtype)
    vacuum=cc[0,0,0];norm=np.sqrt(np.sum(abs(pp)**2)+np.sum(abs(cc-vacuum)**2))
    F=lu_factor(np.load(a.tangent_cache)["following"])
    theta=torch.tensor(np.arccos(1-dt*dt*rank.R**2/2),dtype=ref.dtype)
    lrest=psi_clock(ref[None],-theta)[0];rrest=psi_clock(ref[None],theta)[0]
    def prong(l,r,h):return h*action.SCALE*action.one_form(l,r,h)
    def prong_del(l,m,r):
        return torch.func.grad(lambda v:prong(l,v,dt)+prong(v,r,dt))(m)
    def delta_interval(l,r,h,lam):
        return reduced_frozen_interval(l,r,h)+(lam-1)*prong(l,r,h)
    def exact_interval(l,r,h,lam):
        return action.interval(l,r,h)+(lam-1)*prong(l,r,h)
    def native(l,m,r,lam):
        return reduced_del(l,m,r,dt)+(lam-1)*prong_del(l,m,r)
    def gen(raw,family,local=False):
        v=raw.reshape(5,5,5,4,3);z=torch.zeros_like(v[...,0,:])
        if family==0:g=torch.stack((-v[...,1,:],v[...,0,:],z,z),dim=-2)
        else:g=torch.stack((z,z,-v[...,3,:],v[...,2,:]),dim=-2)
        if local:g=g*mask[...,None,None]
        return g.reshape(-1)
    def measure(l,r,lam):
        d0,d1=torch.func.grad(delta_interval,argnums=(0,1))(l,r,dt,lam)
        ch=torch.stack([torch.dot(d1,gen(r,i)) for i in (0,1)])
        j=torch.stack([torch.dot(d0,gen(l,i,True))+torch.dot(d1,gen(r,i,True)) for i in (0,1)])/dt
        return ch,j
    def solve_linear(l,m,r,rhs):
        def mat(v):
            _,jvp=torch.func.jvp(lambda x:reduced_del(l,m,x,dt),(r,),(torch.tensor(v,dtype=ref.dtype),))
            return lu_solve(F,jvp.detach().numpy())
        op=LinearOperator((r.numel(),r.numel()),matvec=mat,dtype=np.float64)
        b=lu_solve(F,rhs.detach().numpy())
        sol,info=gmres(op,b,rtol=2e-7,atol=0.,restart=25,maxiter=4)
        err=float(np.linalg.norm(mat(sol)-b)/max(np.linalg.norm(b),1e-300))
        if err>2e-5:raise RuntimeError(f"native tangent solve failed: {info} {err}")
        return torch.tensor(sol,dtype=ref.dtype),err
    sp,e1=solve_linear(q,r,s,-prong_del(q,r,s))
    _,dependency=torch.func.jvp(lambda x:reduced_del(r,x,t,dt),(s,),(sp,))
    tp,e2=solve_linear(r,s,t,-prong_del(r,s,t)-dependency)
    v=torch.tensor(1.,dtype=ref.dtype);z=torch.zeros_like(q)
    rows=[]
    for i,(l,m,ld,md) in enumerate(((q,r,z,z),(r,s,z,sp),(s,t,sp,tp))):
        vals,deriv=torch.func.jvp(measure,(l,m,v),(ld,md,v))
        rows.append({"step":i,"current":vals[1].detach().tolist(),"current_coefficient_derivative":deriv[1].detach().tolist(),
                     "charge":vals[0].detach().tolist(),"charge_coefficient_derivative":deriv[0].detach().tolist()})
    currents=np.array([x["current"] for x in rows]);di=np.array([x["current_coefficient_derivative"] for x in rows])
    def solve_candidate(l,m,r,lam):
        hist=[]
        def g(x):return native(l,m,x,lam)
        def resid(x):return lu_solve(F,g(x).detach().numpy())
        for _ in range(6):
            b=resid(r);err=float(np.linalg.norm(b)/norm/dt**2);hist.append(err)
            if err<1e-7:break
            def mat(v):
                _,j=torch.func.jvp(g,(r,),(torch.tensor(v,dtype=ref.dtype),))
                return lu_solve(F,j.detach().numpy())
            op=LinearOperator((r.numel(),r.numel()),matvec=mat,dtype=np.float64)
            delta,flag=gmres(op,-b,rtol=1e-5,atol=0.,restart=25,maxiter=3)
            if flag<0:break
            for j in range(10):
                x=r+torch.tensor(delta*2.**(-j),dtype=ref.dtype)
                if np.linalg.norm(resid(x))<np.linalg.norm(b):r=x;break
            else:break
        full=action.del_residual(l,m,r,dt)+(lam-1)*prong_del(l,m,r)
        defect=float(np.linalg.norm(lu_solve(F,full.detach().numpy()))/norm/dt**2)
        return r,defect,hist
    finite=[]
    for scale in a.scales:
        if scale<=0:raise ValueError("positive kinetic metric scales only")
        ss,e0,h0=solve_candidate(q,r,s+(scale-1)*sp,scale)
        tt,e1_,h1=solve_candidate(r,ss,t+(scale-1)*tp,scale)
        actual=[];charges=[]
        for l,m in ((q,r),(r,ss),(ss,tt)):
            qv,jv=measure(l,m,scale);actual.append(jv.detach().numpy());charges.append(qv.detach().numpy())
        actual=np.asarray(actual);charges=np.asarray(charges)
        predicted=currents+(scale-1)*di
        finite.append({"scale":scale,"exact_candidate_DEL_defects":[e0,e1_],"Newton_traces":[h0,h1],
                       "current_rates":actual.tolist(),"charges":charges.tolist(),
                       "first_current_advance":(actual[1]-actual[0]).tolist(),
                       "second_current_difference":(actual[2]-2*actual[1]+actual[0]).tolist(),
                       "linear_prediction_relative_defect":float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(actual-currents),1.)),
                       "rest_raw_DEL_norm":float(torch.linalg.vector_norm(action.del_residual(lrest,ref,rrest,dt)+(scale-1)*prong_del(lrest,ref,rrest)))})
        print("finite scale",scale,"DEL",e0,e1_,"first and second",finite[-1]["first_current_advance"],finite[-1]["second_current_difference"],flush=True)
    rest_def=float(torch.linalg.vector_norm(prong_del(lrest,ref,rrest)))
    out={"status":"V4_006_NATIVE_PRONG_CLOCK_DRIVE_COEFFICIENT_RESPONSE",
         "action_sha256":sha256(Path(action.__file__)),"state_sha256":sha256(a.archive),
         "rest_cache_sha256":sha256(a.rest_cache),"tangent_cache_sha256":sha256(a.tangent_cache),"dt":dt,
         "native_tangent_solve_relative_defects":[e1,e2],"rest_prong_raw_DEL_norm":rest_def,"rows":rows,
         "baseline_first_current_advance":(currents[1]-currents[0]).tolist(),
         "first_advance_coefficient_derivative":(di[1]-di[0]).tolist(),
         "baseline_second_current_difference":(currents[2]-2*currents[1]+currents[0]).tolist(),
         "second_difference_coefficient_derivative":(di[2]-2*di[1]+di[0]).tolist(),
         "finite_positive_scale_controls":finite,
         "claim_boundary":"Derivative of actual native currents through two implicit roots when varying one existing positive metric coefficient. Fixed initial pair, not an action promotion or physical coefficient selection."}
    print(json.dumps(out,indent=2),flush=True)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")

if __name__=="__main__":main()
