"""Identify which native action carries the phase sum and difference through
the receiving cut. Every term is varied, with no change to the driver."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack,ad_charge,mask_sum
from v4_006_phase_corrected_handoff import native_layers

def main():
    ap=argparse.ArgumentParser()
    for k in ("rest-cache","state","output"):ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--samples",type=int,default=17)
    a=ap.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    rest=np.load(a.rest_cache)["rest"];s=np.load(a.state);dt=float(s["dt"]);n=int(round(float(s["period_updates"])))
    axis=int(s["axis"]);sign=int(s["sign"]);nodes=torch.tensor(s["nodes"],dtype=torch.float64)
    phase=torch.tensor(float(s["phase_per_step"]),dtype=nodes.dtype)
    core=np.linalg.norm(unpack(rest)[0],axis=-1)>1e-12;mask=np.roll(core,sign,axis)&~core
    terms={"core_action":action.incumbent.interval,
           "direct_prong_one_form":lambda l,r,h:h*action.SCALE*action.one_form(l,r,h),
           "matched_six_face_oscillator":lambda l,r,h:action.SCALE*action.face_frame_oscillator_interval(l,r,h),
           "spacetime_plaquette":lambda l,r,h:action.SPACETIME_PLAQUETTE_SCALE*action.spacetime_factorized_face_interval(l,r,h),
           "parallel_material_completion":lambda l,r,h:action.PARALLEL_MATCHED_SCALE*action.parallel_overlap_matched_interval(l,r,h)}
    t=torch.linspace(0,1,a.samples,dtype=nodes.dtype);b,m,f=native_layers(nodes,t,n,phase,axis,sign)
    data={k:[] for k in terms};ident=[]
    qnorm=rank.B*rank.R*np.sum(abs(unpack(rest)[0])**2)*6
    for l,q,r in zip(b,m,f):
        l,q,r=(v.numpy() for v in (l,q,r))
        def terms_at(fn):
            _,rprev=ad_charge(fn,l,q,dt);ln,rnext=ad_charge(fn,q,r,dt)
            return rnext-rprev,ln+rnext
        totals=terms_at(action.interval);values=[]
        for k,fn in terms.items():
            v=terms_at(fn);values.append(v)
            data[k].append(np.asarray([mask_sum(x,mask)/dt for x in v]))
        ident.append(float(np.linalg.norm(np.asarray(totals)-np.sum(values,axis=0))/qnorm))
    out={}
    for k,rows in data.items():
        q,j=np.trapezoid(np.asarray(rows),x=t.numpy()*n*dt,axis=0)
        e=q-j
        out[k]={"receiver_storage":q.tolist(),"receiver_inflow":j.tolist(),"unbalanced":e.tolist(),
                "sum_difference_deficit_half":[float((e[0]+e[1])/2),float((e[0]-e[1])/2)]}
    result={"status":"V4_006_NATIVE_PHASE_SUM_DIFFERENCE_TERM_DECOMPOSITION","action_sha256":sha256(Path(action.__file__)),
            "state_sha256":sha256(a.state),"rest_cache_sha256":sha256(a.rest_cache),"dt":dt,"axis":axis,"sign":sign,"period_updates":n,
            "samples":a.samples,"term_balance_identity_defect":max(ident),"terms":out,
            "claim_boundary":"Exact native local phase variation of each existing action term on the retained collocation screen, not a freely evolved handoff."}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
