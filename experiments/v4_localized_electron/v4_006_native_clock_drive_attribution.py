"""Derive which reciprocal action terms drive the native current beat.

The matched face DEL is linear in the centered defects of Psi and chi,
exactly after separating the wedge's bilinear increment. Invert that
common native quadratic metric, keeping its measured null component
explicit. This attributes the direct Psi current drive to the existing
terms without guessing an action coefficient.
"""
from pathlib import Path
import argparse,json
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import torch
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack,local_contract

def pack(p,c):
    return np.stack((p.real,p.imag,c.real,c.imag),axis=-2).reshape(-1)

def packw(w):
    return np.stack((w.real,w.imag),axis=-2).reshape(-1)

def laplacian(side):
    s=side**3;ix=np.arange(s).reshape((side,)*3);r=np.arange(s)
    L=6*sp.eye(s,format="csr")
    for j in range(3):
        for k in (-1,1):
            L-=sp.csr_matrix((np.ones(s),(r,np.roll(ix,k,j).reshape(-1))),shape=(s,s))
    return L

def wedge_J(p,c):
    blocks=[]
    for p0,c0 in zip(p.reshape(-1,3),c.reshape(-1,3)):
        cols=[]
        for e in np.eye(12):
            v=e[:3]+1j*e[3:6];w=e[6:9]+1j*e[9:]
            dw=(np.cross(c0,v)+np.cross(w,p0))/rank.R
            cols.append(np.r_[dw.real,dw.imag])
        blocks.append(np.stack(cols,axis=1))
    return sp.block_diag(blocks,format="csr")

def main():
    pa=argparse.ArgumentParser()
    for n in ("rest-cache","archive","output"):pa.add_argument("--"+n,type=Path,required=True)
    pa.add_argument("--axis",type=int,default=0);pa.add_argument("--sign",type=int,default=-1)
    pa.add_argument("--transitions",type=int,default=3)
    a=pa.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    ref=np.load(a.rest_cache)["rest"];state=np.load(a.archive);states=state["states"];h=float(state["dt"])
    p0,c0=unpack(ref);core=np.linalg.norm(p0,axis=-1)>1e-12;receiver=np.roll(core,a.sign,a.axis)&~core
    th=np.arccos(1-h*h*rank.R**2/2);co=np.cos(th);si=np.sin(th);u=np.exp(1j*th);rot=u.conjugate()**2;K=2*rank.B*si**2/h**2
    l6=laplacian(5)
    lp=sp.kron(l6,sp.diags([1.]*6+[rank.B]*6),format="csr")
    lw=sp.kron(l6,sp.eye(6),format="csr")
    term={"core":action.incumbent.interval,
          "prong":lambda l,r,h:h*action.SCALE*action.one_form(l,r,h),
          "spacetime":lambda l,r,h:action.SPACETIME_PLAQUETTE_SCALE*action.spacetime_factorized_face_interval(l,r,h),
          "parallel":lambda l,r,h:action.PARALLEL_MATCHED_SCALE*action.parallel_overlap_matched_interval(l,r,h)}
    def force(fn,l,m,r):
        def f(x):return fn(l,x,h)+fn(x,r,h)
        return torch.func.grad(f)(m).detach().numpy()
    def cut(E):
        return np.asarray([np.sum((np.roll(x,1,j)-x)[receiver]) for j,x in enumerate(E)])
    def edge(p,n):
        return np.asarray([u.conjugate()*np.sum(np.roll(p,-1,j).conj()*n-p.conj()*np.roll(n,-1,j),axis=-1) for j in range(3)])
    rows=[]
    # First center is the known q -> r -> s free advance, not reverse launch.
    for idx in range(2,min(len(states)-1,2+a.transitions)):
        l,m,r=states[idx-1:idx+2];pl,cl=unpack(l);p,c=unpack(m);pr,cr=unpack(r)
        rp=pl+pr-2*co*p;rc=cl+cr-2*c;R=pack(rp,rc)
        mixed=(np.cross(cl-c,pl-p)+np.cross(cr-c,pr-p))/rank.R
        J=wedge_J(p,c)
        A=lp+J.T@lw@J
        # Equilibrate the exact metric. Preserve its kernel rather than
        # arbitrarily assigning a response to spatial constant directions.
        d=1/np.sqrt(A.diagonal());scaled=(sp.diags(d)@A@sp.diags(d)).toarray()
        eig,V=eigh(scaled,driver="evd")
        good=eig>max(eig[-1]*1e-10,1e-14);U=V[:,good];lam=eig[good]
        def inv(b):return d*(U@((U.T@(d*b))/lam))
        tensors=[torch.tensor(x,dtype=torch.float64) for x in (l,m,r)]
        f={name:force(fn,*tensors) for name,fn in term.items()}
        full=action.del_residual(*tensors,h).detach().numpy()
        face=force(lambda x,y,h:action.SCALE*action.face_frame_oscillator_interval(x,y,h),*tensors)
        rhs={name:h/rank.B*v for name,v in f.items()}
        rhs["wedge_mixed"]=-J.T@(lw@packw(mixed))
        rhs["numerical_DEL"]=-h/rank.B*full
        Rterms={name:inv(v) for name,v in rhs.items()}
        Rterms["metric_null"]=R-inv(A@R)
        recon=sum(Rterms.values())
        rp0=(p-u.conjugate()*pl)/(2j*si);rn0=(u*pl-p)/(2j*si)
        free=edge(u*rp0,u.conjugate()*rn0)
        def drive(z):
            pp,_=unpack(z);t=pp/(2j*si)
            return cut(edge(u*rp0+t,u.conjugate()*rn0-t)-free)
        direct=drive(R)
        # Exact additive split of a quadratic functional into its linear
        # and symmetric bilinear parts evaluated at the actual total.
        contributions={}
        for name,rr in Rterms.items():
            D=(drive(R+rr)-drive(R-rr)+drive(rr)-drive(-rr))/4
            pp,cc=unpack(rr)
            contributions[name]={"Psi_clock_defect_norm":float(np.linalg.norm(pp)),
                                 "chi_clock_defect_norm":float(np.linalg.norm(cc)),
                                 "drive_real":D.real.tolist(),"drive_imag":D.imag.tolist(),
                                 "direct_current_rate_drive":float(K*D.imag.sum())}
        sumD=sum(np.array(v["drive_real"])+1j*np.array(v["drive_imag"]) for v in contributions.values())
        row={"centered_layer":idx,"metric_rank":int(good.sum()),"metric_dimension":len(eig),"smallest_nonzero_equilibrated_eigenvalue":float(lam[0]),
             "force_partition_relative_error":float(np.linalg.norm(face+sum(f.values())-full)/max(np.linalg.norm(face),1.)),
             "native_face_metric_relative_error":float(np.linalg.norm(-rank.B/h*(A@R+J.T@(lw@packw(mixed)))-face)/max(np.linalg.norm(face),1.)),
             "inverse_defect_reconstruction_relative_error":float(np.linalg.norm(recon-R)/max(np.linalg.norm(R),1e-30)),
             "drive_partition_absolute_error":float(np.linalg.norm(sumD-direct)),
             "direct_drive_real":direct.real.tolist(),"direct_drive_imag":direct.imag.tolist(),
             "direct_current_rate_native_drive":float(K*direct.imag.sum()),"term_contributions":contributions}
        rows.append(row);print("center",idx,"reconstruction",row["inverse_defect_reconstruction_relative_error"],
            "drive",row["direct_current_rate_native_drive"],{name:v["direct_current_rate_drive"] for name,v in contributions.items()},flush=True)
    out={"status":"V4_006_NATIVE_METRIC_CLOCK_DRIVE_ATTRIBUTION","action_sha256":sha256(Path(action.__file__)),
         "rest_cache_sha256":sha256(a.rest_cache),"state_sha256":sha256(a.archive),"dt":h,"axis":a.axis,"sign":a.sign,
         "theta":float(th),"current_prefactor":float(K),"rows":rows,
         "claim_boundary":"Exact native face metric decomposition of actual two-field DEL roots; inverse and null partition are numerical diagnostics, not an installed force or an action correction."}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")

if __name__=="__main__":main()
