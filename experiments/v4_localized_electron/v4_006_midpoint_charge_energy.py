"""Matched two-charge Routhian comparison at the same receiver overlap.

Static seed diagnostics only: the midpoint is not a mandatory dynamical
state. Solve both native global charges, and compare the direct norm-
preserving seed against the geometric face-charge-preserving seed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import root
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_reduced_handoff import rotation
from v4_006_face_charge_transport_seed import bond_norm,pack,unpack

def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--dt',type=float,default=1/256);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank.lock_inertia();torch.set_num_threads(2);dt=a.dt;ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=torch.float64);pr,cr=unpack(ref);vac=cr[0,0,0]
    phase0=torch.tensor([float(np.arccos(1-(dt*rank.R)**2/2)),0.],dtype=ref.dtype)
    def ell(q,ph):return action.interval(q,rotation(q,ph),dt)
    Qmap=torch.func.grad(ell,argnums=1);Q=Qmap(ref,phase0).detach().numpy();scl=np.maximum(abs(Q),1.);ref_r=-(float(ell(ref,phase0))-phase0.numpy()@Q)/dt
    basep=(pr+torch.roll(pr,-1,0))/2;basec=(cr+torch.roll(cr,-1,0))/2;dc=basec-vac
    normalp=basep*torch.linalg.vector_norm(pr)/torch.linalg.vector_norm(basep);qnorm=pack(normalp,basec)
    facep=basep*torch.sqrt(bond_norm(pr)/bond_norm(basep));x=torch.linalg.cross(vac.expand_as(facep),facep,dim=-1);y=torch.linalg.cross(dc,facep,dim=-1)
    target=float(bond_norm(torch.linalg.cross(cr,pr,dim=-1)));aa=float(bond_norm(y));cc=float(bond_norm(x));bb=float(bond_norm(x+y))-aa-cc
    roots=np.roots([aa,bb,cc-target]);b=min((float(z.real) for z in roots if abs(z.imag)<1e-10 and z.real>0),key=lambda z:abs(z-1))
    scale0=float(torch.sqrt(bond_norm(pr)/bond_norm(basep)))
    def geom(v):return pack(float(v[0])*basep,vac+float(v[1])*dc)
    exact=root(lambda v:(Qmap(geom(v),phase0).detach().numpy()-Q)/scl,[scale0,b],method='hybr',options={'xtol':1e-10,'maxfev':35});qgeo=geom(exact.x)
    rows=[]
    for name,q in [('Psi_norm_preserved',qnorm),('face_charge_preserved',qgeo)]:
        initial=Qmap(q,phase0).detach().numpy();fun=lambda v:(Qmap(q,torch.tensor(v,dtype=ref.dtype)).detach().numpy()-Q)/scl
        solved=root(fun,phase0.numpy(),method='hybr',options={'xtol':1e-10,'maxfev':45});ph=torch.tensor(solved.x,dtype=ref.dtype)
        defect=float(np.linalg.norm(fun(solved.x)));r=-(float(ell(q,ph))-solved.x@Q)/dt-ref_r
        p_,c_=unpack(q);rows.append({'seed':name,'charge_relative_at_rest_clock':((initial-Q)/scl).tolist(),'phase_root':solved.x.tolist(),'native_charge_root_defect':defect,
            'fixed_two_charge_Routhian_delta':r,'relative_to_rest_Routhian':r/ref_r,'Psi_norm_ratio':float(torch.linalg.vector_norm(p_)/torch.linalg.vector_norm(pr)),
            'chi_contrast_norm_ratio':float(torch.linalg.vector_norm(c_-vac)/torch.linalg.vector_norm(cr-vac))})
    d={'status':'V4_006_MATCHED_MIDPOINT_TWO_CHARGE_DIAGNOSTIC','action_sha256':sha256(Path(action.__file__)),'dt':dt,'phase_charge_rest':Q.tolist(),'Routhian_rest':ref_r,'rows':rows,
       'claim_boundary':'Exact two-charge Routh reduction restricted to two diagnostic midpoint seeds. This is not the optimal physical barrier or a native trajectory.'}
    a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(d,indent=2))

if __name__=='__main__':main()
