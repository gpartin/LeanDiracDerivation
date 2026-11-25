"""Integrate native conserved-charge phase rates into a single orbit chart.

The phase momenta are solved from the actual action on each incoming and
outgoing trial interval. A real periodic Fourier phase function plus a mean
advance is obtained by solving its *finite differences*, not resetting the
phase at every update. Physically consistent relative-periodic phase data
must also close the remaining full native DEL and independent holdouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve

import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_charge_reduced_handoff import rotation
from v4_006_current_action_relative_orbit import shift
from v4_006_phase_corrected_handoff import charge,evaluate_fourier


def basis(times,order):
    t=2*np.pi*np.asarray(times)[:,None]*np.arange(1,order+1)[None,:]
    return np.column_stack((np.cos(t)-1,np.sin(t)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--phase-roots',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);state=np.load(a.state);nodes=torch.tensor(state['nodes'],dtype=torch.float64)
    dt=float(state['dt']);n=int(round(float(state['period_updates'])));h=1/n;axis=int(state['axis']);sign=int(state['sign']);t=np.arange(nodes.shape[0])/nodes.shape[0];order=(nodes.shape[0]-1)//2
    source=json.loads(a.phase_roots.read_text());base_phase=np.asarray(source['phase_reference']);Q=np.asarray(source['native_rest_charge']);qs=max(abs(Q[0]),1e-300)
    if abs(base_phase[0]-float(state['phase_per_step']))>1e-12 or source['period_updates']!=n:raise ValueError('phase roots and state differ')
    roots_in=np.asarray([r['incoming_phase'] for r in source['rows']]);roots_out=np.asarray([r['outgoing_phase'] for r in source['rows']])
    # The exact interval increments are finite differences of one periodic
    # phase, including the free mean phase per native step. Solve both the
    # past and future interval data in a common gauge phi(0)=0.
    system=np.vstack((np.column_stack((np.ones(t.size),basis(t,order)-basis(t-h,order))),
                         np.column_stack((np.ones(t.size),basis(t+h,order)-basis(t,order)))))
    rhs=np.vstack((roots_in-base_phase,roots_out-base_phase));coeff,_,rank_ls,_=np.linalg.lstsq(system,rhs,rcond=1e-12)
    ref=torch.tensor(np.load(a.rest_cache)['rest'],dtype=nodes.dtype);vac=ref.reshape(125,12)[0].clone();vac[:6]=0;norm=float(torch.linalg.vector_norm(ref-vac.repeat(125)))
    factor=lu_factor(np.load(a.tangent_cache)['following'])
    def q_at(tt):
        times=torch.tensor(tt,dtype=nodes.dtype);return shift(evaluate_fourier(nodes,times),sign*times,axis)
    def sample(tt,corrected):
        left=q_at(tt-h);current=q_at(tt);right=q_at(tt+h);rows=[]
        theta_in=np.broadcast_to(base_phase,(len(tt),2)).copy();theta_out=theta_in.copy()
        if corrected:
            theta_in+=coeff[0]+(basis(tt,order)-basis(tt-h,order))@coeff[1:]
            theta_out+=coeff[0]+(basis(tt+h,order)-basis(tt,order))@coeff[1:]
        for j,x in enumerate(tt):
            before=rotation(left[j],-torch.tensor(theta_in[j],dtype=nodes.dtype));after=rotation(right[j],torch.tensor(theta_out[j],dtype=nodes.dtype));mid=current[j]
            raw=action.del_residual(before,mid,after,dt).detach().numpy();r=lu_solve(factor,raw)/dt**2
            qi=charge(before,mid,dt).detach().numpy();qo=charge(mid,after,dt).detach().numpy()
            rows.append({'fraction':float(x),'acceleration_defect':float(np.linalg.norm(r)/norm),'charge_in_relative_defect':((qi-Q)/qs).tolist(),
                'charge_out_relative_defect':((qo-Q)/qs).tolist(),'incoming_phase':theta_in[j].tolist(),'outgoing_phase':theta_out[j].tolist()})
        return rows
    rows={}
    for name,tt in [('collocation',t),('holdout',t+.5/t.size)]:
        baseline=sample(tt,False);corrected=sample(tt,True)
        def summary(rs):return {'max_acceleration_defect':max(r['acceleration_defect'] for r in rs),'max_charge_defect':float(max(np.max(np.abs(r[k])) for r in rs for k in ('charge_in_relative_defect','charge_out_relative_defect')))}
        rows[name]={'baseline_summary':summary(baseline),'corrected_summary':summary(corrected),'baseline':baseline,'corrected':corrected}
        print(name,rows[name]['baseline_summary'],rows[name]['corrected_summary'],flush=True)
    result={'status':'V4_006_INTEGRATED_NATIVE_PHASE_MOMENTUM_CHART','action_sha256':sha256(Path(action.__file__)),'state_sha256':sha256(a.state),'phase_root_sha256':sha256(a.phase_roots),
            'dt':dt,'period_updates':n,'mean_phase_step_correction':coeff[0].tolist(),'total_phase_over_handoff':(n*(base_phase+coeff[0])).tolist(),
            'periodic_phase_coefficients':coeff[1:].tolist(),'phase_difference_matrix_rank':int(rank_ls),'maximum_interval_phase_fit_error':float(np.max(abs(system@coeff-rhs))),
            'tests':rows,'claim_boundary':'Integrated cyclic phase chart from native phase charges only. A charge-consistent phase chart is not a native orbit unless the full DEL vanishes on all layers and free release succeeds.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
