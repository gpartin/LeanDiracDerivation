"""Cheap necessary tests before a full rebuild of material-frame action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor,lu_solve

import canonical_transfer_oscillator_spectrum as spectrum
import cube_incidence_extended_packet_gate as dirac
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_bond_action as incumbent
import v4_006_material_face_action as candidate
from v4_003_fixed_phase_space_readout import sha256
from v4_006_phase_corrected_handoff import native_layers, unpack


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['Psi','W','chi','Psi_W','all']);a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    state=np.load(a.state);nodes=torch.tensor(state['nodes'],dtype=torch.float64);dt=float(state['dt']);n=int(round(float(state['period_updates'])));phase=torch.tensor(float(state['phase_per_step']),dtype=nodes.dtype)
    axis=int(state['axis']);sign=int(state['sign']);rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=nodes.dtype);vac=ref.reshape(125,12)[0].clone();vac[:6]=0;norm=float(torch.linalg.vector_norm(ref-vac.repeat(125)))
    d=np.load(a.tangent_cache);P,C,F=(d[k] for k in ('previous','current','following'));factor=lu_factor(F)
    theta=float(np.arccos(1-(dt*rank.R)**2/2));rotm=spectrum.rotation_matrix(rest.size,-theta);rotp=spectrum.rotation_matrix(rest.size,theta)
    rest_layers=tuple(torch.tensor(x,dtype=nodes.dtype) for x in (rotm@rest,rest,rotp@rest));base_rest=incumbent.del_residual(*rest_layers,dt)
    times=torch.arange(nodes.shape[0],dtype=nodes.dtype)/nodes.shape[0];before,current,after=native_layers(nodes,times,n,phase,axis,sign)
    base=[incumbent.del_residual(before[i],current[i],after[i],dt).detach().numpy() for i in range(len(times))]
    psi_frame,chi_frame,sites=envelope.carrier_frames(rest);cases=[]
    for copy,basis in envelope.flavor_bases().items():
        alpha,beta=dirac.compressed_clifford(basis);raw=envelope.raw_columns(basis,psi_frame,chi_frame,sites,dirac.OFFSETS,1/np.sqrt(rank.B))
        for s in (-1,1):
            prev,now,following,_=dirac.packet(0,s,alpha,beta,dt,centered_time=True,frequency_branch='negative')
            triple=tuple(raw@dirac.real_coordinates(x) for x in (prev,now,following));pv,cv,fv=rotm@triple[0],triple[1],rotp@triple[2]
            native=P@pv+C@cv+F@fv;cases.append((copy,s,pv,cv,fv,native))
    def scaled(x):return lu_solve(factor,x)/dt**2
    rows=[]
    for mode in a.modes:
        rest_delta=candidate.delta_del(*rest_layers,dt,mode).detach().numpy();rr=base_rest.detach().numpy()+rest_delta
        rr_norm=float(np.linalg.norm(rr));rr_acc=float(np.linalg.norm(scaled(rr))/norm);path=[]
        for i,t in enumerate(times):
            correction=candidate.delta_del(before[i],current[i],after[i],dt,mode).detach().numpy()
            d0=scaled(base[i]);d1=scaled(base[i]+correction);dd=scaled(correction)
            path.append({'fraction':float(t),'incumbent_defect':float(np.linalg.norm(d0)/norm),'candidate_defect':float(np.linalg.norm(d1)/norm),
                         'correction_response_fraction_against_defect':float(np.dot(-d0,dd)/max(np.dot(d0,d0),1e-300))})
        small=[]
        for copy,s,pv,cv,fv,basev in cases:
            def fn(p,q,r):return candidate.delta_del(p,q,r,dt,mode)
            _,dv=torch.func.jvp(fn,rest_layers,tuple(torch.tensor(x,dtype=nodes.dtype) for x in (pv,cv,fv)))
            d0=scaled(basev);dd=scaled(dv.detach().numpy());small.append({'copy':copy,'sign':s,'incumbent_acceleration_defect':float(np.linalg.norm(d0)/np.linalg.norm(cv)),
                'candidate_acceleration_defect_in_incumbent_units':float(np.linalg.norm(d0+dd)/np.linalg.norm(cv)),
                'response_fraction_against_defect':float(np.dot(-d0,dd)/max(np.dot(d0,d0),1e-300))})
        row={'mode':mode,'rest_raw_DEL_norm':rr_norm,'rest_preconditioned_acceleration_defect':rr_acc,'incumbent_rest_raw_DEL_norm':float(torch.linalg.vector_norm(base_rest)),
             'path':path,'small_momentum_probes':small,'maximum_path_incumbent_defect':max(r['incumbent_defect'] for r in path),'maximum_path_candidate_defect':max(r['candidate_defect'] for r in path),
             'maximum_small_momentum_incumbent_defect':max(r['incumbent_acceleration_defect'] for r in small),'maximum_small_momentum_candidate_defect':max(r['candidate_acceleration_defect_in_incumbent_units'] for r in small)}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('path','small_momentum_probes')},indent=2),flush=True)
    result={'status':'V4_006_MATERIAL_FACE_DERIVATIVE_ACTION_PROBE_NOT_ADMITTED','action_sha256':sha256(Path(incumbent.__file__)),'candidate_sha256':sha256(Path(candidate.__file__)),
            'state_sha256':sha256(a.state),'tangent_cache_sha256':sha256(a.tangent_cache),'dt':dt,'rows':rows,
            'claim_boundary':'Native reciprocal action replacements only, evaluated on matched rest and handoff data. Residual preconditioning uses the unchanged action for comparison, not a claim that the changed action evolves those data.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
