"""Recover the prior vacuum-activated chi coupling on the exact V4 action.

At the rest state s_i s_j (P_i-P_j)=0 on every directed C19 bond,
where s=(rho_chi-rho_Psi)/19^2. Thus the exact Hessian is J^T J of
the weighted square residual. No direction, carrier label or Dirac matrix
enters this candidate potential. A phase-sensitive field-gradient variant
is tested separately because the dyad cannot see chi's complex phase.
Both are research probes, not changes to the incumbent action.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as spectrum
import cube_incidence_extended_packet_gate as dirac
import edge_snap_action_rank as rank
import localized_stability as native
import relative_kinetic_probe as physical
from v4_003_fixed_phase_space_readout import sha256
from v4_003_moving_native_graph_audit import (
    HERE, measure_graph, native_graph, orth, sparse, spatial_lifts, spatial_shapes)


def residual_jacobian(rest, kind, jacobian=True):
    p,c=physical.unpack_raw(rest,5)
    points=np.arange(125).reshape((5,)*3)
    p=p.reshape(-1,3); c=c.reshape(-1,3)
    s=(np.sum(abs(c)**2,axis=1)-np.sum(abs(p)**2,axis=1))/19**2
    ds=2*np.column_stack((-p.real,-p.imag,c.real,c.imag))/19**2
    unit=np.column_stack((np.eye(3),1j*np.eye(3))).reshape(3,6)
    if kind=='dyad':
        value=c[:,:,None]*c[:,None,:].conj()
        dvalue=np.zeros((125,3,3,12),complex)
        for k in range(6):
            d=unit[:,k]
            dvalue[...,6+k]=d[None,:,None]*c[:,None,:].conj()+c[:,:,None]*d[None,None,:].conj()
        value=value.reshape(125,9); dvalue=dvalue.reshape(125,9,12)
        normalizer=19
    else:
        value=c; dvalue=np.zeros((125,3,12),complex);dvalue[...,6:]=unit
        normalizer=1
    rows=[]; cols=[]; values=[]; residuals=[]; row=0
    for offset,w in native.DIRECTIONS:
        neighbors=np.roll(points,tuple(-x for x in offset),axis=(0,1,2)).ravel()
        for i,j in enumerate(neighbors):
            scale=np.sqrt(w/2)/normalizer
            delta=value[i]-value[j]
            z=scale*s[i]*s[j]*delta
            residuals.extend(np.r_[z.real,z.imag])
            if not jacobian:continue
            di=scale*s[j]*(delta[:,None]*ds[i]+s[i]*dvalue[i])
            dj=scale*s[i]*(delta[:,None]*ds[j]-s[j]*dvalue[j])
            for idx,mat in ((i,di),(j,dj)):
                block=np.vstack((mat.real,mat.imag))
                rr,cc=np.nonzero(block)
                rows.extend(rr+row);cols.extend(cc+12*idx);values.extend(block[rr,cc])
            row+=2*len(z)
    res=np.asarray(residuals)
    if not jacobian:return res
    return res,coo_matrix((values,(rows,cols)),shape=(row,rest.size)).tocsc()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--tangent-cache',type=Path,required=True);parser.add_argument('--rest-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cycles',type=int,default=10)
    args=parser.parse_args();rank.lock_inertia();dt=args.dt
    rest=np.load(args.rest_cache)['rest'];data=np.load(args.tangent_cache)
    p,c,f=(data[k] for k in ('previous','current','following'))
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    P=sparse(p@spectrum.rotation_matrix(rest.size,-theta));factor=splu(sparse(f@spectrum.rotation_matrix(rest.size,theta)))
    C=sparse(c);lifts,_,bases=spatial_lifts(rest);raw=lifts['contrast'];shapes=spatial_shapes()
    q,_=orth(np.column_stack([raw[k]@shapes for k in ('plus','minus')]))
    seeds=[];cases=[]
    for copy in ('plus','minus'):
        alpha,beta=dirac.compressed_clifford(bases[copy])
        for axis in range(4):
            for sign in (-1,1):
                _,field,_,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                seeds.append(raw[copy]@dirac.real_coordinates(field));cases.append((copy,axis,sign))
    seed=np.column_stack(seeds);zero=np.zeros_like(q);backward=factor.solve(-(P@q));rows=[];checks=[]
    for kind in ('dyad','complex_field'):
        res,J=residual_jacobian(rest,kind);H=J.T@J
        rng=np.random.default_rng(739);v=rng.normal(size=rest.size);v/=np.linalg.norm(v);eps=1e-5
        fd=(residual_jacobian(rest+eps*v,kind,False)-residual_jacobian(rest-eps*v,kind,False))/(2*eps)
        checks.append({'kind':kind,'rest_square_norm':float(np.linalg.norm(res)),
                       'rest_gradient_norm':float(np.linalg.norm(J.T@res)),
                       'jacobian_relative_defect':float(np.linalg.norm(fd-J@v)/np.linalg.norm(fd))})
        if np.linalg.norm(res)>1e-10:raise ValueError('not rest-null')
        for strength in ((0,1,19,1197) if kind=='dyad' else (1,19,1197)):
            # exact interval variation of -dt*(U(left)+U(right))/2.
            CT=C-dt*strength*H
            def advance(a,b):return factor.solve(-(CT@a+P@b))
            forward=advance(q,zero);g,diag=native_graph(q,forward,backward)
            row={'kind':kind,'strength':strength,'native_graph':diag}
            if g is not None:
                graph,_=measure_graph(q,forward,backward,g,dt)
                now=seed.copy();before=q@g@(q.T@now);initial=np.sqrt(np.sum(now**2+((now-before)/dt)**2,axis=0));maxratio=np.ones(len(cases))
                for _ in range(int(np.ceil(args.cycles*2*np.pi/(19*dt)))):
                    after=advance(now,before);before,now=now,after
                    ratio=np.sqrt(np.sum(now**2+((now-before)/dt)**2,axis=0))/initial
                    maxratio=np.maximum(maxratio,ratio)
                row.update(graph=graph,maximum_velocity_aware_norm_ratio=float(np.max(maxratio)),worst_case=cases[int(np.argmax(maxratio))])
            rows.append(row);print(json.dumps(row),flush=True)
    result={'status':'RECIPROCAL_VACUUM_ORIENTATION_TANGENT_PROBE_NOT_ADMITTED','baseline_action_sha256':sha256(HERE/'live_clifford_bond_action.py'),
            'baseline_tangent_sha256':sha256(args.tangent_cache),'rest_sha256':sha256(args.rest_cache),'dt':dt,'cycles':args.cycles,'checks':checks,'rows':rows,
            'claim_boundary':'Rest-null reciprocal C19 potential differences, exact Hessians at zero square residual, diagnostic strengths only. No change to incumbent action or physical claim.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
