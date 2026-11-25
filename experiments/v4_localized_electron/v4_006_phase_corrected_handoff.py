"""Free-period nonlinear collocation of the frozen, phase-corrected V4 action.

The exact relative-periodic boundary is imposed only as a coordinate chart.
Neither the Dirac comparator nor a translated intermediate target enters DEL.
The eight initial core assignments remain exact; the exterior and every
other collocation layer are unknown. A period is solved in a declared finite
window, rounded to integer native steps, and tested on independent time
nodes. This is a boundary-value screen, never a free evolution certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_current_action_relative_orbit import shift, psi_clock


HERE = Path(__file__).resolve().parent
SIDE = 5


def evaluate_fourier(nodes, times):
    count = nodes.shape[0]
    modes = torch.fft.fftfreq(count, d=1/count, dtype=nodes.dtype)
    weights = torch.exp(2j*torch.pi*times[:, None]*modes[None, :])
    return (weights @ torch.fft.fft(nodes, dim=0)).real/count


def unpack(raw):
    values = raw.reshape(-1, SIDE, SIDE, SIDE, 4, 3)
    return torch.complex(values[..., 0, :], values[..., 1, :]), torch.complex(values[..., 2, :], values[..., 3, :])


def charge(left, right, dt):
    momentum = torch.func.grad(action.interval, argnums=1)(left, right, dt)
    values = right.reshape(SIDE, SIDE, SIDE, 4, 3)
    zero = torch.zeros_like(values[..., 0, :])
    pgen = torch.stack((-values[..., 1, :], values[..., 0, :], zero, zero), dim=-2).reshape(-1)
    cgen = torch.stack((zero, zero, -values[..., 3, :], values[..., 2, :]), dim=-2).reshape(-1)
    return torch.stack((torch.dot(momentum, pgen), torch.dot(momentum, cgen)))


def native_layers(nodes, at, n, phase, axis, sign):
    """Chart with q(t+T)=phase_total * tau q(t), both native layers."""
    h = 1/n
    before = shift(evaluate_fourier(nodes, at-h), sign*(at-h), axis)
    current = shift(evaluate_fourier(nodes, at), sign*at, axis)
    after = shift(evaluate_fourier(nodes, at+h), sign*(at+h), axis)
    return psi_clock(before, -phase), current, psi_clock(after, phase)


def field_metrics(raw, ref, mask):
    p, c = unpack(raw); rp, rc = unpack(ref)
    vacuum = rc[0, 0, 0, 0].clone(); dc, rdc = c-vacuum, rc-vacuum
    psq = torch.sum(abs(p)**2); csq = torch.sum(abs(dc)**2)
    rp2 = torch.sum(abs(rp)**2); rc2 = torch.sum(abs(rdc)**2)
    return {'Psi_core_fidelity': float((abs(torch.sum(rp.conj()*p))**2/torch.clamp(psq*rp2,min=1e-300)).detach()),
            'chi_contrast_fidelity': float((abs(torch.sum(rdc.conj()*dc))**2/torch.clamp(csq*rc2,min=1e-300)).detach()),
            'Psi_exterior_fraction': float((torch.sum(abs(p[:,~mask])**2)/torch.clamp(psq,min=1e-300)).detach()),
            'chi_exterior_relative_norm': float((torch.linalg.vector_norm(dc[:,~mask])/torch.sqrt(rc2)).detach()),
            'Psi_norm_relative_to_rest': float(torch.sqrt(psq/rp2).detach()),
            'chi_contrast_norm_relative_to_rest': float(torch.sqrt(csq/rc2).detach())}


def face_bond_norm(q):
    result=0
    for a in (1,2,3):
        d=torch.roll(q,-1,a)-q
        result=result+torch.sum(torch.real(d.conj()*d),dim=(1,2,3,4))
    return result


def project_face_capacity(samples,ref):
    """BVP coordinate chart, not an update rule: preserve two face norms."""
    p,c=unpack(samples);rp,rc=unpack(ref);vac=rc[0,0,0,0];dc=c-vac
    pn=face_bond_norm(rp)[0];wn=face_bond_norm(torch.linalg.cross(rc,rp,dim=-1))[0]
    p=p*torch.sqrt(pn/face_bond_norm(p))[:,None,None,None,None]
    x=torch.linalg.cross(vac.expand_as(p),p,dim=-1);y=torch.linalg.cross(dc,p,dim=-1)
    aa=face_bond_norm(y);cc=face_bond_norm(x);bb=face_bond_norm(x+y)-aa-cc
    discr=bb**2-4*aa*(cc-wn)
    if bool(torch.any(discr.detach()<=0)) or bool(torch.any(aa.detach()<=0)):raise ValueError('outside real face-capacity chart')
    b=(-bb+torch.sqrt(discr))/(2*aa)
    c=vac+b[:,None,None,None,None]*dc
    return torch.stack((p.real,p.imag,c.real,c.imag),dim=-2).reshape(samples.shape)


def reduced_frozen_interval(left,right,dt):
    # At the frozen unit matched weight, the shared base cancels exactly:
    # L_base + (L_overlap - L_base) = L_overlap. Preserve all native terms.
    return (action.overlap_material.interval(left,right,dt)
            +dt*action.SCALE*action.one_form(left,right,dt)
            +action.SCALE*action.face_frame_oscillator_interval(left,right,dt)
            +action.SPACETIME_PLAQUETTE_SCALE*action.spacetime_factorized_face_interval(left,right,dt)
            -dt*action.PARALLEL_MATCHED_POTENTIAL_SCALE*(action.parallel_overlap_potential(left)+action.parallel_overlap_potential(right))/2)


def reduced_del(previous,current,following,dt):
    def summed(p,q,r):return reduced_frozen_interval(p,q,dt)+reduced_frozen_interval(q,r,dt)
    return torch.func.grad(summed,argnums=1)(previous,current,following)


def main():
    p=argparse.ArgumentParser();p.add_argument('--rest-cache',type=Path,required=True);p.add_argument('--tangent-cache',type=Path,required=True)
    p.add_argument('--dt',type=float,default=1/256);p.add_argument('--reference-updates',type=int,default=17564)
    p.add_argument('--nodes',type=int,default=5);p.add_argument('--axis',type=int,choices=(0,1,2),default=0);p.add_argument('--sign',type=int,choices=(-1,1),default=-1)
    p.add_argument('--iterations',type=int,default=50);p.add_argument('--period-window',type=float,default=2.0)
    p.add_argument('--fixed-period-updates',type=int,help='comparison control at an independently known crossing period')
    p.add_argument('--seed',choices=('fourier','local'),default='local');p.add_argument('--warm-start',type=Path)
    p.add_argument('--project-face-capacity',action='store_true');p.add_argument('--fast-exact-reduction',action='store_true')
    p.add_argument('--phase-chart',choices=('all_coordinates','site','prong'),default='all_coordinates')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--state-output',type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2);dt=a.dt
    if a.nodes<3 or a.nodes%2!=1:raise ValueError('odd temporal node count required')
    if a.period_window<=1:raise ValueError('period_window must exceed 1')
    config={'link':action.LINK_SOURCE,'scale':action.SCALE,'spacetime_scale':action.SPACETIME_PLAQUETTE_SCALE,
            'parallel_matched_scale':action.PARALLEL_MATCHED_SCALE,'parallel_clock_potential_scale':action.PARALLEL_MATCHED_POTENTIAL_SCALE,
            'optional_scales':[action.PLAQUETTE_SCALE,action.SYMPLECTIC_PLAQUETTE_SCALE,action.PARALLEL_OVERLAP_SCALE,action.PARALLEL_FACE_SCALE,action.TRIANGLE_DELTA_SCALE,action.MATERIAL_AREA_SCALE],
            'chi_payload':action.CHI_PAYLOAD}
    if config != {'link':'joint_face_laplacian_plus_frame_oscillator','scale':1197.0,'spacetime_scale':6/361,'parallel_matched_scale':1.,'parallel_clock_potential_scale':1.,'optional_scales':[0.]*6,'chi_payload':1.}:raise ValueError(f'unexpected action {config}')
    rest=np.load(a.rest_cache)['rest'];ref=torch.tensor(rest,dtype=torch.float64)
    rawshape=ref.reshape(SIDE,SIDE,SIDE,12);vacuum=rawshape[0,0,0].clone();vacuum[:6]=0
    contrast=ref-vacuum.repeat(SIDE**3);norm=torch.linalg.vector_norm(contrast)
    core=torch.sum(rawshape[...,:6]**2,dim=-1)>1e-12
    exterior_mask=(~core)[...,None].expand(SIDE,SIDE,SIDE,12).reshape(-1)
    F=np.load(a.tangent_cache)['following'];factor=lu_factor(F)
    preconditioner=torch.tensor(lu_solve(factor,np.eye(rest.size)),dtype=torch.float64)
    theta=float(np.arccos(1-dt**2*native_time.CHI0**2/2));times=torch.arange(a.nodes,dtype=torch.float64)/a.nodes
    refcharge=charge(psi_clock(ref[None,:],torch.tensor(-theta,dtype=ref.dtype))[0],ref,dt)
    # Initial guesses only: Fourier chart of the rest seed, or a localized
    # interpolation between adjacent native configurations. All interior
    # samples are released during the solve; neither path is a pass rule.
    if a.seed=='local':
        translated=torch.roll(rawshape,a.sign,a.axis).reshape(-1)
        mix=(1-torch.cos(torch.pi*times))/2
        seed=shift((1-mix[:,None])*ref+mix[:,None]*translated,-a.sign*times,a.axis)
    else:seed=ref.repeat(a.nodes,1)
    if a.warm_start:
        old=np.load(a.warm_start);seed=evaluate_fourier(torch.tensor(old['nodes'],dtype=torch.float64),times).detach()
        initial_n=float(old['period_updates']);initial_phase=float(old['phase_per_step'])
    else:initial_n=float(a.reference_updates);initial_phase=theta
    # Exact starting core without fixing the surrounding native precursor.
    exterior=torch.nn.Parameter(seed[0,exterior_mask]-ref[exterior_mask]);inner=torch.nn.Parameter(seed[1:].clone())
    phase_shape=(a.nodes-1,SIDE,SIDE,SIDE,2,1 if a.phase_chart=='site' else 3)
    phase_parameters=torch.nn.Parameter(torch.zeros(phase_shape,dtype=ref.dtype))
    initial_physical=shift(seed[1:],a.sign*times[1:],a.axis).detach()
    period_raw=torch.nn.Parameter(torch.tensor(float(np.arctanh(np.clip(np.log(initial_n/a.reference_updates)/np.log(a.period_window),-.999,.999))),dtype=torch.float64))
    phase_delta=torch.nn.Parameter(torch.tensor((initial_phase-theta)*initial_n,dtype=torch.float64))

    def data():
        q0=ref.clone();q0[exterior_mask]+=exterior
        if a.phase_chart=='all_coordinates':intermediate=inner
        else:
            pp,cc=unpack(initial_physical)
            pp=pp*torch.exp(1j*phase_parameters[...,0,:]);cc=cc*torch.exp(1j*phase_parameters[...,1,:])
            trial=torch.stack((pp.real,pp.imag,cc.real,cc.imag),dim=-2).reshape(initial_physical.shape)
            intermediate=shift(trial,-a.sign*times[1:],a.axis)
        nodes=torch.cat((q0[None,:],intermediate),dim=0)
        if a.project_face_capacity:
            # Do not alter the exact initial core. Intermediate variables
            # use a charge-capacity chart only within this boundary solve.
            physical_nodes=shift(nodes[1:],a.sign*times[1:],a.axis)
            nodes=torch.cat((nodes[:1],shift(project_face_capacity(physical_nodes,ref),-a.sign*times[1:],a.axis)),dim=0)
        n=(torch.tensor(float(a.fixed_period_updates),dtype=ref.dtype) if a.fixed_period_updates
           else a.reference_updates*torch.exp(np.log(a.period_window)*torch.tanh(period_raw)))
        phase=theta+phase_delta/n
        return nodes,n,phase

    def evaluate(at, rounded=False, exact=False):
        nodes,n,phase=data()
        if rounded:
            n=torch.round(n);phase=theta+phase_delta/n
        before,current,after=native_layers(nodes,at,n,phase,a.axis,a.sign)
        fn=action.del_residual if exact or not a.fast_exact_reduction else reduced_del
        r=torch.stack([fn(before[i],current[i],after[i],dt) for i in range(at.numel())])
        acceleration=r@preconditioner.T/dt**2
        return acceleration,r,before,current,after,n,phase

    # Charge is not substituted for DEL. The conserved rest-carrier phase
    # charges are checked separately after the solve, so an apparent vacuum
    # or wrong-charge root cannot pass by shrinking its action residual.
    def objective():
        acceleration,*_=evaluate(times)
        return torch.sum(acceleration**2)/(a.nodes*norm**2)

    initial=float(torch.sqrt(objective()).detach());trace=[]
    reduction_check=None
    if a.fast_exact_reduction:
        _,_,bp,cp,fp,_,_=evaluate(times)
        full=action.del_residual(bp[1],cp[1],fp[1],dt);reduced=reduced_del(bp[1],cp[1],fp[1],dt)
        reduction_check=float(torch.linalg.vector_norm(full-reduced).detach()/max(float(torch.linalg.vector_norm(full).detach()),1e-300))
        if reduction_check>1e-8:raise ValueError(f'exact interval reduction mismatch: {reduction_check}')
    optimizer=torch.optim.LBFGS([exterior,inner if a.phase_chart=='all_coordinates' else phase_parameters,period_raw,phase_delta],lr=.7,max_iter=a.iterations,max_eval=a.iterations*2,
                                 tolerance_grad=1e-11,tolerance_change=1e-15,history_size=25,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=objective();loss.backward();trace.append(float(torch.sqrt(loss.detach())))
        if len(trace)%10==0:print('DEL evaluation',len(trace),'defect',trace[-1],flush=True)
        return loss
    optimizer.step(closure)
    nodes,n,phase=data();nvalue=float(n.detach());rounded=int(round(nvalue));phase_value=float(phase.detach())
    holdout=times+0.5/a.nodes
    accel,raw,before,current,after,_,_=evaluate(times,True,True)
    test,test_raw,_,_,_,_,_=evaluate(holdout,True,True)
    charges=[];energy=[]
    for i in range(a.nodes):
        charges.append(charge(before[i].detach(),current[i].detach(),dt).detach().numpy())
        from v4_002_discrete_energy_gate import interval_energy
        energy.append(interval_energy(before[i].detach().numpy(),current[i].detach().numpy(),dt,2.5e-5))
    charges=np.asarray(charges);refq=refcharge.detach().numpy();energy=np.asarray(energy)
    pernode=[{'fraction':float(t),'defect':float(torch.linalg.vector_norm(z).detach()/norm)} for t,z in zip(times,accel)]
    worst=int(np.argmax([x['defect'] for x in pernode]));v=accel[worst].detach();vp,vc=unpack(v)
    peak_power=float(torch.sum(abs(vp)**2+abs(vc)**2));
    result={'status':'V4_006_PHASE_CORRECTED_FREE_PERIOD_COLLOCATION_SCREEN','action_sha256':sha256(Path(action.__file__)),'action_configuration':config,
            'rest_raw_sha256':hashlib.sha256(rest.tobytes()).hexdigest(),'tangent_cache_sha256':sha256(a.tangent_cache),'dt':dt,'axis':a.axis,'sign':a.sign,'nodes':a.nodes,'seed':a.seed,
            'period_window_updates':[a.reference_updates/a.period_window,a.reference_updates*a.period_window],
            'fixed_comparator_period_control':a.fixed_period_updates,'solved_continuous_period_updates':nvalue,'rounded_period_updates':rounded,
            'rounded_crossing_time':rounded*dt,'solved_speed_cells_per_time':a.sign/(rounded*dt),'phase_per_step':phase_value,'reference_phase_per_step':theta,
            'initial_acceleration_defect':initial,'final_acceleration_defect':float(torch.linalg.vector_norm(accel).detach()/(np.sqrt(a.nodes)*norm)),
            'holdout_acceleration_defect':float(torch.linalg.vector_norm(test).detach()/(np.sqrt(a.nodes)*norm)),'final_raw_DEL_RMS':float(torch.sqrt(torch.mean(raw**2)).detach()),
            'holdout_raw_DEL_RMS':float(torch.sqrt(torch.mean(test_raw**2)).detach()),'per_node':pernode,
            'worst_node_Psi_residual_power_fraction':float(torch.sum(abs(vp)**2)/max(peak_power,1e-300)),
            'initial_metrics':field_metrics(current[0],ref,core),'charge_reference':refq.tolist(),'charge_samples':charges.tolist(),
            'maximum_charge_defect_over_reference_Psi_charge':(np.max(abs(charges-refq),axis=0)/max(abs(refq[0]),1e-300)).tolist(),
            'interval_energy_samples':energy.tolist(),'maximum_relative_energy_variation':float(np.max(abs(energy-energy[0]))/max(np.max(abs(energy)),1e-300)),
            'optimizer_evaluations':len(trace),'trace':trace,'initial_core_fixed':True,'initial_exterior_free':True,'runtime_forcing':False,
            'project_face_capacity_BVP_chart':a.project_face_capacity,'phase_chart':a.phase_chart,'exact_reduction_DEL_check':reduction_check,
            'claim_boundary':'Finite collocation with free exterior, interior and crossing time. Endpoint chart is exact, but native DEL and independent holdout residual must vanish before a free-release test. Not a pass or a no-go.'}
    np.savez_compressed(a.state_output,nodes=nodes.detach().numpy(),period_updates=nvalue,phase_per_step=phase_value,dt=dt,axis=a.axis,sign=a.sign)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('trace','charge_samples','interval_energy_samples')},indent=2))


if __name__=='__main__':main()
