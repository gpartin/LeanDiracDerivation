"""Revalidate V4-003 on the frozen V4-002 action, with no Dirac launch.

Two spatial lifts are compared: the old companion-only lift and the exact
carrier-minus-vacuum contrast found in the prior V4 research.  The same
finite packet space contains both copies, all component directions, signed
momentum envelopes and the seven compact source/face-neighbor centers.
Only the native recurrence determines its previous-layer graph. Dirac is
used afterwards to evaluate its generator on the declared packet samples.
No projection is applied to the native evolution.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as dirac
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import relative_kinetic_probe as physical
from v4_003_fixed_phase_space_readout import sha256


HERE = Path(__file__).resolve().parent
SIDE = 5
OFFSETS = list(itertools.product(range(-2, 3), repeat=3))
FIELDS = ("plus", "minus")


def relative(actual, expected):
    return float(np.linalg.norm(actual-expected) /
                 max(np.linalg.norm(expected), 1e-300))


def orth(matrix):
    left, values, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = values > values[0]*1e-10
    return left[:, keep], values


def sparse(matrix):
    matrix = matrix.copy()
    matrix[np.abs(matrix) < 1e-11] = 0
    return csc_matrix(matrix)


def spatial_shapes():
    points = np.asarray(OFFSETS).reshape(SIDE, SIDE, SIDE, 3)
    gaussian = np.exp(-np.sum(points**2, axis=-1)/(2*1.35**2))
    # Native geometry first; none of these columns uses a Dirac eigenvector.
    momenta = [np.zeros(3)] + [
        sign*.35*np.eye(3)[axis] for axis in range(3) for sign in (-1, 1)]
    momenta += [sign*.35*np.ones(3)/np.sqrt(3) for sign in (-1, 1)]
    shapes = [gaussian*np.exp(1j*np.einsum('...a,a->...',points,k))
              for k in momenta]
    for offset in [(0,0,0)] + [tuple(sign if dim == axis else 0
            for dim in range(3)) for axis in range(3) for sign in (-1,1)]:
        delta = np.zeros((SIDE,)*3, complex)
        delta[tuple(2+x for x in offset)] = 1
        shapes.append(delta)
    cols=[]
    for shape in shapes:
        for component in range(4):
            field=np.zeros((SIDE,SIDE,SIDE,4),complex)
            field[...,component]=shape
            cols.extend([dirac.real_coordinates(field),dirac.real_coordinates(1j*field)])
    return np.column_stack(cols)


def spatial_lifts(rest):
    psi_frame,chi_frame,sites=envelope.carrier_frames(rest)
    bases=envelope.flavor_bases()
    legacy={k:envelope.raw_columns(bases[k],psi_frame,chi_frame,sites,
                  OFFSETS,complex(1/np.sqrt(native_time.B),0)) for k in FIELDS}
    # The rest cache stores *physical* chi, not mass-normalized coordinates.
    # localized_stability.unpack divides chi by sqrt(B), and importing an old
    # diagnostic can even set its B to 229. Use the raw inverse of rank.pack.
    psi,chi=physical.unpack_raw(rest,SIDE)
    if not np.array_equal(rank.pack(psi,chi),rest):
        raise ValueError('rest field coordinates do not round trip')
    psi_profile=np.stack([psi[s] for s in sites])
    chi_profile=np.stack([chi[s] for s in sites])
    exterior=[s for s in itertools.product(range(SIDE),repeat=3) if s not in sites]
    vacuum=chi[exterior[0]]
    if abs(np.linalg.norm(vacuum)-native_time.CHI0)>1e-10:
        raise ValueError('vacuum chi is not in physical units')
    if np.max(np.abs(np.stack([chi[s] for s in exterior])-vacuum))>1e-10:
        raise ValueError('rest cache has nonuniform exterior chi')
    # Same native amplitude unit is applied to both fields, so translation
    # carries the actual physical contrast, not an independently scaled chi.
    amplitude=float(np.linalg.norm(psi_profile[0]))
    contrast={k:envelope.raw_columns(bases[k],psi_profile/amplitude,
                  (chi_profile-vacuum)/amplitude,sites,OFFSETS,1+0j) for k in FIELDS}
    shift=[]
    for axis in range(3):
        dpsi=(np.roll(psi,1,axis=axis)-np.roll(psi,-1,axis=axis))/2
        dchi=(np.roll(chi,1,axis=axis)-np.roll(chi,-1,axis=axis))/2
        v=rank.pack(dpsi,dchi)
        shift.append(v/np.linalg.norm(v))
    return dict(legacy=legacy,contrast=contrast),np.column_stack(shift),bases


def get_action(dt, tangent, rest):
    data=np.load(tangent)
    p,c,f=(data[n] for n in ('previous','current','following'))
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotate_m=native_spectrum.rotation_matrix(rest.size,-theta)
    rotate_p=native_spectrum.rotation_matrix(rest.size,theta)
    factor=splu(sparse(f@rotate_p))
    P=sparse(p@rotate_m)
    C=sparse(c)
    return lambda current,previous: factor.solve(-(C@current+P@previous))


def native_graph(q,forward,backward):
    r=q.shape[1]
    a=q.T@forward
    b=q.T@backward
    M=np.block([[a,b],[np.eye(r),np.zeros((r,r))]])
    eig=np.linalg.eigvals(M)
    distances=np.sort(abs(eig-1))
    cut=(distances[r-1]+distances[r])/2
    margin=float(distances[r]-distances[r-1])
    _,z,count=schur(M,output='real',sort=lambda re,im: abs(complex(re,im)-1)<cut)
    if count != r:
        return None,{'native_selected_dimension':int(count),'expected_dimension':r,
                     'clock_branch_gap':margin,'branch_not_separated':True}
    top,bottom=z[:r,:r],z[r:,:r]
    cond=float(np.linalg.cond(top))
    if not np.isfinite(cond) or cond>1e10:
        return None,{'native_selected_dimension':int(count),'expected_dimension':r,
                     'clock_branch_gap':margin,'top_graph_condition':cond}
    # The selected native branch is the half of the projected recurrence
    # closest to the co-rotating rest multiplier 1.  This is an explicit
    # spectral hypothesis, not an imposed Dirac phase. Validate its full
    # native cross block and comparator before accepting it.
    g=np.linalg.solve(top.T,bottom.T).T
    return g,{'native_selected_dimension':int(count),'expected_dimension':r,
              'clock_branch_gap':margin,'top_graph_condition':cond}


def measure_graph(q,forward,backward,g,dt):
    top=forward+backward@g
    gen=q.T@top
    phase=np.vstack((q,q@g))
    next_phase=np.vstack((top,q))
    residual=next_phase-phase@gen
    current_residual=top-q@gen
    return {'full_phase_invariance_relative':relative(next_phase,phase@gen),
            'configuration_cross_block_relative':float(np.linalg.norm(current_residual)/np.linalg.norm(top)),
            'cross_block_per_dt_squared':float(np.linalg.norm(current_residual)/np.linalg.norm(q)/dt**2),
            'recurrence_graph_defect_per_dt_squared':float(np.linalg.norm(np.eye(g.shape[0])-g@gen)/np.sqrt(g.shape[0])/dt**2),
            'previous_graph_condition':float(np.linalg.cond(g)),
            'phase_condition':float(np.linalg.cond(phase)),
            'retained_spectral_radius':float(np.max(abs(np.linalg.eigvals(gen))))},phase


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dt',type=float,required=True)
    p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--rest-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();dt=args.dt
    rank.lock_inertia()
    rest=np.load(args.rest_cache)['rest']
    advance=get_action(dt,args.tangent_cache,rest)
    lifts,translation,bases=spatial_lifts(rest)
    shapes=spatial_shapes()
    rows=[]
    for name,raws in lifts.items():
        separate={k:raws[k]@shapes for k in FIELDS}
        q,singular=orth(np.column_stack(list(separate.values())))
        ranks={k:orth(separate[k])[0].shape[1] for k in FIELDS}
        full_q={k:orth(raws[k])[0] for k in FIELDS}
        # Shared spatial image is a kinematic obstruction, not a dynamic one.
        overlap=float(np.linalg.norm(full_q['plus']-full_q['minus']@(full_q['minus'].T@full_q['plus']))/np.sqrt(full_q['plus'].shape[1]))
        row={'lift':name,'packet_configuration_rank':int(q.shape[1]),'individual_packet_ranks':ranks,
             'joint_rank_deficit':int(sum(ranks.values())-q.shape[1]),
             'all_center_plus_outside_minus_fraction':overlap,
             'raw_translation_defects':[float(np.linalg.norm(v-q@(q.T@v))) for v in translation.T]}
        zero=np.zeros_like(q); forward=advance(q,zero); backward=advance(zero,q)
        # Reproduce the V4-003 least-squares graph, only on this enlarged
        # frozen space; include the near-rest solution when leakage leaves
        # null directions (minimum change from G=I).
        A=backward-q@(q.T@backward); D=forward-q@(q.T@forward)
        delta,*_=np.linalg.lstsq(A,-D-A,rcond=1e-8)
        graph=np.eye(q.shape[1])+delta
        g_metrics,phase=measure_graph(q,forward,backward,graph,dt)
        row['native_leakage_graph']=g_metrics
        graph,diagnostics=native_graph(q,forward,backward)
        row['native_spectral_branch']=diagnostics
        oracle=[]
        # This is a necessary consistency diagnostic, never a launch used to
        # certify movement. If the same exact lift intertwined Dirac and the
        # native recurrence, the second difference here would agree.
        for k in FIELDS:
            alpha,beta=dirac.compressed_clifford(bases[k])
            for axis in range(4):
                for sign in (-1,1):
                    dp,dc,dn,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                    prev,now,target=[raws[k]@dirac.real_coordinates(x) for x in (dp,dc,dn)]
                    following=advance(now,prev)
                    discrepancy=(following-target)/dt**2
                    expected=(target-2*now+prev)/dt**2
                    epsi,echi=physical.unpack_raw(discrepancy,SIDE)
                    pp=float(np.linalg.norm(epsi)**2); cp=float(np.linalg.norm(echi)**2)
                    oracle.append({'copy':k,'axis':axis,'sign':sign,
                        'native_acceleration_deficit_over_state_norm':float(np.linalg.norm(discrepancy)/np.linalg.norm(now)),
                        'native_acceleration_relative_to_Dirac_acceleration':float(np.linalg.norm(discrepancy)/max(np.linalg.norm(expected),1e-300)),
                        'Psi_power_fraction':pp/max(pp+cp,1e-300),
                        'chi_power_fraction':cp/max(pp+cp,1e-300),
                        'chi_normal_power_fraction':float(np.linalg.norm(echi[...,2])**2/max(pp+cp,1e-300))})
        row['Dirac_recurrence_consistency_diagnostic']=oracle
        row['maximum_acceleration_deficit_over_state_norm']=max(c['native_acceleration_deficit_over_state_norm'] for c in oracle)
        if graph is not None:
            metrics,phase=measure_graph(q,forward,backward,graph,dt)
            row['native_spectral_graph']=metrics
            checks=[]
            for k in FIELDS:
                alpha,beta=dirac.compressed_clifford(bases[k])
                for axis in range(4):
                    for sign in (-1,1):
                        dp,dc,dn,_=dirac.packet(axis,sign,alpha,beta,dt,centered_time=True,frequency_branch='negative')
                        p0=raws[k]@dirac.real_coordinates(dc)
                        z=q.T@p0
                        native_prev=q@graph@z
                        native_next=advance(p0,native_prev)
                        native_dirac_prev=raws[k]@dirac.real_coordinates(dp)
                        reference=raws[k]@dirac.real_coordinates(dn)
                        expected_rate=(reference-native_dirac_prev)/(2*dt)
                        actual_rate=(native_next-native_prev)/(2*dt)
                        checks.append({'copy':k,'axis':axis,'sign':sign,
                          'initial_span_relative_error':relative(q@z,p0),
                          'previous_layer_difference_per_dt':relative(native_prev,native_dirac_prev)/dt,
                          'native_Dirac_centered_rate_relative_error':relative(actual_rate,expected_rate),
                          'native_Dirac_step_relative_error':relative(native_next,reference)})
            row['independent_comparator']=checks
            row['maximum_rate_relative_error']=max(c['native_Dirac_centered_rate_relative_error'] for c in checks)
        rows.append(row)
        print(name, {k:v for k,v in row.items() if k not in ('independent_comparator','Dirac_recurrence_consistency_diagnostic')},flush=True)
    result={'stage':'V4-003','status':'MOVING_NATIVE_GRAPH_AUDIT','dt':dt,
            'action_sha256':sha256(HERE/'live_clifford_bond_action.py'),
            'tangent_cache_sha256':sha256(args.tangent_cache),
            'rest_cache_sha256':sha256(args.rest_cache),
            'native_graph_selection':'near-rest spectral half of native compressed recurrence, checked by unrestricted native recurrence',
            'rows':rows,'action_changed':False,'runtime_projection':False,
            'claim_boundary':'A finite packet-space diagnostic, not a full nonlinear moving orbit or a stage pass.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
