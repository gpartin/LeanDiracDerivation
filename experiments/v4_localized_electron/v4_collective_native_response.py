"""Native frequency residual with collective, not microscopic, constraints.

If Q is an orthonormal collective chart, fix only Q* x = a and allow every
complementary native Ψ/χ coordinate to respond. Complete Q to [Q,N]. The
minimum native DEL acceleration residual is the least-squares problem
min_y ||D(z)(Q a + N y)||. No inverse requirement enters the native action.
Finite-box, fixed-frequency numerical test; not a no-go for nonlinear hops.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr, svd
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as spectrum
import cube_incidence_extended_packet_gate as dirac
import edge_snap_action_rank as rank
from v4_003_fixed_phase_space_readout import sha256
from v4_003_moving_native_graph_audit import HERE, orth, sparse, spatial_lifts, spatial_shapes
from v4_003_native_chi_response import complexified


def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True);p.add_argument('--rest-cache',type=Path,required=True)
    p.add_argument('--tangent-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank.lock_inertia();dt=a.dt;rest=np.load(a.rest_cache)['rest'];data=np.load(a.tangent_cache)
    lifts,_,bases=spatial_lifts(rest);raw=lifts['contrast'];shape=spatial_shapes();source=np.column_stack([raw[k]@shape for k in ('plus','minus')]);
    q_joint,_=orth(source)
    psi_ids=np.arange(rest.size).reshape(125,12)[:,:6].ravel();source_psi=np.zeros_like(source);source_psi[psi_ids]=source[psi_ids]
    q_psi,_=orth(source_psi)
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2));mass=dirac.COMPARATOR_MASS
    P=sparse(data['previous']@spectrum.rotation_matrix(rest.size,-theta));C=sparse(data['current']);F=sparse(data['following']@spectrum.rotation_matrix(rest.size,theta));update=splu(F.astype(complex))
    k0=np.array([2*np.pi/5,0,0]);momentum=dirac.c19_first_symbol(k0);energy=float(np.sqrt(mass*mass+momentum@momentum));omega=float((np.arccos(1-dt**2*energy**2/2)-theta)/dt)
    z=np.exp(1j*omega*dt);D=update.solve((P/z+C+F*z).toarray())/dt**2
    points=np.asarray(list(itertools.product(range(-2,3),repeat=3))).reshape(5,5,5,3)
    targets=[];labels=[]
    for axis in range(3):
        for sign in (-1,1):
            k=sign*2*np.pi/5*np.eye(3)[axis]
            for copy,basis in bases.items():
                al,be=dirac.compressed_clifford(basis);h=mass*be
                for i,value in enumerate(dirac.c19_first_symbol(k)):h=h+value*al[i]
                eigen,vectors=np.linalg.eigh(h);wave=np.exp(1j*np.einsum('...a,a->...',points,k))
                for band in np.where(eigen<0)[0]:
                    targets.append(complexified(wave[...,None]*vectors[:,band],raw[copy]));labels.append({'copy':copy,'axis':axis,'sign':sign,'band':int(band)})
    targets=np.column_stack(targets);rows=[]
    for name,q in [('joint_contrast_coordinates',q_joint),('Psi_collective_only',q_psi)]:
        complete,_=qr(q,mode='full');N=complete[:,q.shape[1]:];base=q@(q.T@targets);rhs=-D@base
        dn=D@N;left,singular,right=svd(dn,full_matrices=False,lapack_driver='gesdd')
        for rtol in (1e-8,1e-10,1e-12):
            keep=singular>singular[0]*rtol
            correction=N@(right[keep].conj().T@((left[:,keep].conj().T@rhs)/singular[keep,None]))
            solved=base+correction;constraint=q.T@(solved-targets);residual=D@solved
            denom=np.maximum(np.linalg.norm(base,axis=0),1e-300)
            cases=[dict(label,original_acceleration_defect=float(v),minimum_acceleration_defect=float(w),relative_native_correction=float(c),collective_constraint_defect=float(e)) for label,v,w,c,e in zip(labels,np.linalg.norm(D@targets,axis=0)/denom,np.linalg.norm(residual,axis=0)/denom,np.linalg.norm(correction,axis=0)/denom,np.linalg.norm(constraint,axis=0)/denom)]
            row={'chart':name,'collective_dimension':int(q.shape[1]),'free_dimension':int(N.shape[1]),'rtol':rtol,'numerical_rank':int(np.sum(keep)),
                 'max_acceleration_defect':max(c['minimum_acceleration_defect'] for c in cases),'max_relative_correction':max(c['relative_native_correction'] for c in cases),
                 'max_constraint_defect':max(c['collective_constraint_defect'] for c in cases),'cases':cases}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
    result={'status':'NATIVE_COLLECTIVE_CONSTRAINED_RESPONSE_DIAGNOSTIC','dt':dt,'co_rotating_frequency':omega,'action_sha256':sha256(HERE/'live_clifford_bond_action.py'),
            'tangent_cache_sha256':sha256(a.tangent_cache),'rest_cache_sha256':sha256(a.rest_cache),'rows':rows,'claim_boundary':'Least-squares residual at one nonzero discrete momentum magnitude in a declared collective chart. Not a moving graph, proof of invariance, or a nonlinear no-go.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
