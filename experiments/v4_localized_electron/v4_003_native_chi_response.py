"""Eliminate live chi by its own discrete variational equation.

This is a frequency-domain inverse diagnostic of the unchanged V4-002 action,
not an equation or initial-state modification. At each C19 probe momentum,
hold the Psi amplitude of one independently specified Dirac mode, solve every
chi coordinate (and, in a second control, Psi's third component) from the
native DEL equation, and measure the remaining Psi equation. All coordinates,
including exterior chi, remain in the physical units of the action cache.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import splu

import cube_incidence_extended_packet_gate as dirac
import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as spectrum
import edge_snap_action_rank as rank
from v4_003_moving_native_graph_audit import spatial_lifts, sparse, SIDE, HERE
from v4_003_fixed_phase_space_readout import sha256


def complexified(field, lift):
    # Real representation of the complex spinor and its i-multiple.
    # This embeds the e^{+i omega t} component into the complexification
    # of the native real phase-space equations.
    return lift@dirac.real_coordinates(field)-1j*lift@dirac.real_coordinates(1j*field)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--rest-cache',type=Path,required=True)
    parser.add_argument('--tangent-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();dt=args.dt
    rank.lock_inertia()
    rest=np.load(args.rest_cache)['rest']
    lifts,_,bases=spatial_lifts(rest)
    data=np.load(args.tangent_cache)
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    P=sparse(data['previous']@spectrum.rotation_matrix(rest.size,-theta))
    C=sparse(data['current'])
    F=sparse(data['following']@spectrum.rotation_matrix(rest.size,theta))
    update=splu(F.astype(np.complex128))
    indices=np.arange(rest.size).reshape(SIDE,SIDE,SIDE,4,3)
    psi_parallel=indices[...,:2,:2].ravel()
    psi_normal=indices[...,:2,2].ravel()
    chi=indices[...,2:,:].ravel()
    coords=np.asarray(list(itertools.product(range(-2,3),repeat=3))).reshape(SIDE,SIDE,SIDE,3)
    rows=[]
    for momenta in [[np.zeros(3)], [sign*2*np.pi/SIDE*np.eye(3)[axis]
                                         for axis in range(3) for sign in (-1,1)]]:
        # All members of the cubic orbit have the same C19 energy.
        tr=dirac.c19_first_symbol(momenta[0]);m=dirac.COMPARATOR_MASS
        energy=float(np.sqrt(m*m+tr@tr))
        omega=float((np.arccos(1-dt**2*energy**2/2)-theta)/dt)
        z=np.exp(1j*omega*dt)
        M=P/z+C+F*z
        fixed_sets={'chi_only':np.sort(np.r_[psi_parallel,psi_normal]),
                    'chi_and_normal_Psi':np.sort(psi_parallel)}
        for label,fixed in fixed_sets.items():
            free=np.setdiff1d(np.arange(rest.size),fixed)
            mm=M[free,:][:,free].toarray()
            mf=M[free,:][:,fixed].toarray()
            # Absolute DEL terms scale strongly with the frame inertia. Use
            # rank-aware minimum-norm solve and report the unsolved residual.
            inverse,eff_rank=pinv(mm,rtol=1e-10,return_rank=True)
            for k in momenta:
                for copy,basis in bases.items():
                    alpha,beta=dirac.compressed_clifford(basis)
                    h=m*beta
                    for axis,v in enumerate(dirac.c19_first_symbol(k)):
                        h=h+v*alpha[axis]
                    vals,vecs=np.linalg.eigh(h)
                    selection=np.where(vals<0)[0]
                    wave=np.exp(1j*np.einsum('...a,a->...',coords,k))
                    for band in selection:
                        field=wave[...,None]*vecs[:,band]
                        prescribed=complexified(field,lifts['legacy'][copy])
                        old=complexified(field,lifts['contrast'][copy])
                        rhs=-(mf@prescribed[fixed])
                        solved=np.zeros(rest.size,complex)
                        solved[fixed]=prescribed[fixed]
                        solved[free]=inverse@rhs
                        defect=M@solved
                        delta=update.solve(defect)/dt**2
                        oracle_delta=update.solve(M@old)/dt**2
                        norm=max(np.linalg.norm(solved[fixed]),1e-300)
                        rows.append({'free_fields':label,'copy':copy,
                           'momentum':k.tolist(),'band':int(band), 'co_rotating_frequency':omega,
                           'free_operator_rank':int(eff_rank),'free_operator_dimension':int(len(free)),
                           'free_DEL_relative_residual':float(np.linalg.norm(defect[free])/max(np.linalg.norm(rhs),1e-300)),
                           'full_acceleration_deficit_over_prescribed_Psi_norm':float(np.linalg.norm(delta)/norm),
                           'Psi_acceleration_deficit_over_prescribed_Psi_norm':float(np.linalg.norm(delta[np.r_[psi_parallel,psi_normal]])/norm),
                           'previous_naive_contrast_acceleration_deficit_over_Psi_norm':float(np.linalg.norm(oracle_delta)/norm),
                           'native_chi_response_over_Psi_norm':float(np.linalg.norm(solved[chi])/norm),
                           'native_chi_normal_response_over_Psi_norm':float(np.linalg.norm(solved[indices[...,2:,2].ravel()])/norm)})
            print(label,'omega',omega,'rank',eff_rank,'of',len(free),flush=True)
    result={'status':'V4_003_NATIVE_CHI_SCHUR_DIAGNOSTIC','dt':dt,
            'action_sha256':sha256(HERE/'live_clifford_bond_action.py'),
            'tangent_cache_sha256':sha256(args.tangent_cache),
            'rest_cache_sha256':sha256(args.rest_cache),'rows':rows,
            'claim_boundary':'Dirac frequency probes are inverse requirements only; no native graph, action change, movement or pass is inferred.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
