"""Exact oriented phase balance and mixed Psi-chi curvature of the frozen action.

Each axis' phase storage minus its signed inflow is the negative local
phase contraction of its DEL force. The wedge's finite-layer clock defect
splits into the two fields' clock defects and a bilinear increment. All
diagnostics use the same action and native layers. Native release solves
the actual unrestricted implicit step, not the candidate path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack, dotimag, local_contract, ad_charge, mask_sum
from v4_006_phase_corrected_handoff import native_layers, reduced_del

HERE = Path(__file__).resolve().parent


def lap(f, axis):
    return 2*f - np.roll(f, 1, axis) - np.roll(f, -1, axis)


def oriented_terms(left, middle, right, h):
    p0,c0 = unpack(left)
    p,c = unpack(middle)
    p1,c1 = unpack(right)
    w0,w,w1 = (np.cross(c0,p0)/rank.R, np.cross(c,p)/rank.R, np.cross(c1,p1)/rank.R)
    cos = 1 - h*h*rank.R**2/2
    rp = p0 + p1 - 2*cos*p
    rc = c0 + c1 - 2*c
    rw = w0 + w1 - 2*cos*w
    wp = np.cross(c,rp)/rank.R
    wc = np.cross(rc,p)/rank.R
    wm = (np.cross(c0-c,p0-p)+np.cross(c1-c,p1-p))/rank.R
    pieces = {"psi": [], "chi": [], "w_psi": [], "w_chi": [], "w_mixed": []}
    storage, inflow, exact = [], [], []
    for a in range(3):
        def phase(f, r, coeff): return coeff/h*dotimag(f, lap(r,a))
        pieces["psi"].append(phase(p,rp,rank.B))
        pieces["chi"].append(phase(c,rc,rank.B**2))
        pieces["w_psi"].append(phase(w,wp,rank.B))
        pieces["w_chi"].append(phase(w,wc,rank.B))
        pieces["w_mixed"].append(phase(w,wm,rank.B))
        def interval(l,r,k,cc):
            dl=np.roll(l,-1,a)-l
            dr=np.roll(r,-1,a)-r
            ml=k/h*(cc*dl-dr)
            mr=k/h*(cc*dr-dl)
            ql=-dotimag(l,ml)+np.roll(dotimag(np.roll(l,-1,a),ml),1,a)
            qr=-dotimag(r,mr)+np.roll(dotimag(np.roll(r,-1,a),mr),1,a)
            return ql, qr
        def field(l,m,r,k,cc):
            qprev=interval(l,m,k,cc)[1]
            ql,qr=interval(m,r,k,cc)
            return qr-qprev,ql+qr
        ps,pi=field(p0,p,p1,rank.B,cos)
        ws,wi=field(w0,w,w1,rank.B,cos)
        cs,ci=field(c0,c,c1,rank.B**2,1)
        storage.append(np.stack((ps+ws,cs+ws)))
        inflow.append(np.stack((pi+wi,ci+wi)))
        wphase=phase(w,rw,rank.B)
        exact.append(np.stack((pieces["psi"][-1]+wphase,pieces["chi"][-1]+wphase)))
    pieces={k:np.asarray(v) for k,v in pieces.items()}
    def split(k): return np.stack((pieces[k],pieces[k]),axis=1)
    z=np.zeros_like(pieces["psi"])
    source={"psi":np.stack((pieces["psi"],z),axis=1),
            "chi":np.stack((z,pieces["chi"]),axis=1),
            **{k:split(k) for k in ("w_psi","w_chi","w_mixed")}}
    return np.asarray(storage),np.asarray(inflow),np.asarray(exact),source,float(np.linalg.norm(rw-wp-wc-wm))


def axis_sum(arr,mask):
    return arr[:,:,mask].sum(axis=2)


def phase_step(prev, now, following, dt, factor, scale, iters, backward=False):
    """Unrestricted DEL Newton solve with rest preconditioner only."""
    trace=[]
    def raw(x):
        return reduced_del(x,now,prev,dt) if backward else reduced_del(prev,now,x,dt)
    def residual(x):
        return lu_solve(factor,raw(x).detach().numpy())
    f=following.clone()
    for k in range(iters):
        r=residual(f)
        err=np.linalg.norm(r)/scale/dt**2
        trace.append(float(err))
        if err<1e-7: break
        # A local matrix-free Jacobian, never an adjustment of the action.
        if k==0:
            delta=-r
        else:
            def matvec(v):
                _,jvp=torch.func.jvp(raw,(f,),(torch.tensor(v,dtype=f.dtype),))
                return lu_solve(factor,jvp.detach().numpy())
            op=LinearOperator((f.numel(),)*2,matvec=matvec,dtype=np.float64)
            delta,info=gmres(op,-r,rtol=1e-5,atol=0.,restart=20,maxiter=2)
            if info<0: break
        for j in range(12):
            trial=f+torch.tensor(delta*2.**(-j),dtype=f.dtype)
            if np.linalg.norm(residual(trial))<np.linalg.norm(r):
                f=trial
                break
        else: break
    exact=(action.del_residual(f,now,prev,dt) if backward else action.del_residual(prev,now,f,dt)).detach().numpy()
    err=float(np.linalg.norm(lu_solve(factor,exact))/scale/dt**2)
    return f,trace,err


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rest-cache",type=Path,required=True)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--tangent-cache",type=Path)
    ap.add_argument("--samples",type=int,default=33)
    ap.add_argument("--release-fractions",type=float,nargs="*",default=[])
    ap.add_argument("--release-steps",type=int,default=3)
    ap.add_argument("--newton-iterations",type=int,default=5)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    rank.lock_inertia()
    torch.set_num_threads(2)
    state=np.load(a.state)
    rest=np.load(a.rest_cache)["rest"]
    dt=float(state["dt"]); n=int(round(float(state["period_updates"])))
    axis=int(state["axis"]); sign=int(state["sign"])
    nodes=torch.tensor(state["nodes"],dtype=torch.float64)
    phase=torch.tensor(float(state["phase_per_step"]),dtype=nodes.dtype)
    rp,rc=unpack(rest)
    core=np.linalg.norm(rp,axis=-1)>1e-12
    receiver=np.roll(core,sign,axis)&~core
    sender=core&~np.roll(core,sign,axis)
    vacuum=rc[0,0,0]
    scale=np.sqrt(np.sum(abs(rp)**2)+np.sum(abs(rc-vacuum)**2))
    qscale=rank.B*np.sin(np.arccos(1-dt*dt*rank.R**2/2))/dt*np.sum(abs(rp)**2)*6
    def inspect(l,q,r,include_full=True):
        ds,div,e,parts,werr=oriented_terms(l,q,r,dt)
        s=ds.sum(axis=0);j=div.sum(axis=0);f=e.sum(axis=0)
        data={"wedge_clock_chain_defect":werr,
              "orientation_balance_identity_defect":float(np.linalg.norm(ds-div-e)/qscale),
              "orientation_shared_field_decomposition_defect":float(np.linalg.norm(e-sum(parts.values()))/qscale),
              "receiver_storage_rate_by_axis":(axis_sum(ds,receiver)/dt).tolist(),
              "receiver_inflow_rate_by_axis":(axis_sum(div,receiver)/dt).tolist(),
              "receiver_axis_exchange_required_by_storage_minus_inflow":(axis_sum(e,receiver)/dt).tolist(),
              "receiver_axis_exchange_components":{k:(axis_sum(v,receiver)/dt).tolist() for k,v in parts.items()},
              "sender_axis_exchange_required_by_storage_minus_inflow":(axis_sum(e,sender)/dt).tolist(),
              "global_axis_exchange":(e.sum(axis=(2,3,4))/dt).tolist()}
        if include_full:
            ql0,qr0=ad_charge(action.interval,l,q,dt)
            ql1,qr1=ad_charge(action.interval,q,r,dt)
            full=(qr1-qr0)-(ql1+qr1)
            defect=full-f
            raw=action.del_residual(torch.tensor(l),torch.tensor(q),torch.tensor(r),dt).detach().numpy()
            ff=lambda l,r,h:action.SCALE*action.face_frame_oscillator_interval(l,r,h)
            def ft(l,m,r):
                return ff(l,m,dt)+ff(m,r,dt)
            face_del=torch.func.grad(ft,argnums=1)(torch.tensor(l),torch.tensor(q),torch.tensor(r)).detach().numpy()
            data.update({"receiver_other_action_exchange":(mask_sum(defect,receiver)/dt).tolist(),
                         "receiver_full_unbalanced_exchange":(mask_sum(full,receiver)/dt).tolist(),
                         "receiver_full_inflow_rate":(mask_sum(ql1+qr1,receiver)/dt).tolist(),
                         "receiver_full_storage_rate":(mask_sum(qr1-qr0,receiver)/dt).tolist(),
                         "full_phase_balance_identity_defect":float(np.linalg.norm(full+local_contract(q,raw))/qscale),
                         "face_phase_balance_ad_defect":float(np.linalg.norm(f+local_contract(q,face_del))/qscale),
                         "full_LOCAL_balance_norm_over_rest_charge":float(np.linalg.norm(full)/qscale)})
        return data
    times=torch.linspace(0,1,a.samples,dtype=nodes.dtype)
    b,q,r=native_layers(nodes,times,n,phase,axis,sign)
    rows=[]
    for t,l,m,f in zip(times,b,q,r):
        data=inspect(l.numpy(),m.numpy(),f.numpy())
        data["fraction"]=float(t)
        rows.append(data)
    def integral(xs): return np.trapezoid(np.asarray(xs),x=times.numpy()*n*dt,axis=0).tolist()
    totals={}
    for k in ("receiver_storage_rate_by_axis","receiver_inflow_rate_by_axis","receiver_axis_exchange_required_by_storage_minus_inflow",
              "receiver_other_action_exchange","receiver_full_unbalanced_exchange","receiver_full_inflow_rate","receiver_full_storage_rate"):
        totals[k]=integral([x[k] for x in rows])
    totals["receiver_axis_exchange_components"]={k:integral([x["receiver_axis_exchange_components"][k] for x in rows])
                                                 for k in rows[0]["receiver_axis_exchange_components"]}
    release=[]
    if a.release_fractions:
        if not a.tangent_cache: raise ValueError("native release requires tangent cache")
        factor=lu_factor(np.load(a.tangent_cache)["following"])
        for t in a.release_fractions:
            bs,ms,fs=native_layers(nodes,torch.tensor([t],dtype=nodes.dtype),n,phase,axis,sign)
            prev,now,trial=bs[0],ms[0],fs[0]
            desired=inspect(prev.numpy(),now.numpy(),trial.numpy())
            target_total=np.asarray(desired["receiver_full_storage_rate"])
            steps=[]
            for step in range(a.release_steps):
                if step: trial=2*now-prev
                nxt,trace,err=phase_step(prev,now,trial,dt,factor,scale,a.newton_iterations)
                d=inspect(prev.numpy(),now.numpy(),nxt.numpy())
                d.update({"step":step+1,"newton_trace":trace,"exact_full_DEL_defect":err,
                          "receiver_storage_rate_relative_to_path":(np.asarray(d["receiver_full_storage_rate"])/np.where(abs(target_total)>1e-30,target_total,1)).tolist()})
                steps.append(d)
                print("release",t,step+1,"DEL",err,"path rate ratios",d["receiver_storage_rate_relative_to_path"],flush=True)
                if err>1e-6: break
                prev,now=now,nxt
            release.append({"fraction":t,"desired_path":desired,"steps":steps})
    result={"status":"V4_006_ORIENTED_NATIVE_PHASE_EXCHANGE_DIAGNOSTIC",
            "action_sha256":sha256(Path(action.__file__)),"state_sha256":sha256(a.state),"rest_cache_sha256":sha256(a.rest_cache),
            "tangent_cache_sha256":sha256(a.tangent_cache) if a.tangent_cache else None,
            "dt":dt,"axis":axis,"sign":sign,"period_updates":n,"samples":a.samples,
            "integrated_receiver_terms":totals,"rows":rows,"native_release":release,
            "claim_boundary":"Exact face-orientation DEL phase identities plus candidate-path quadrature and unrestricted local native release, not a full-cell trajectory."}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(totals,indent=2),flush=True)


if __name__=="__main__":
    main()
