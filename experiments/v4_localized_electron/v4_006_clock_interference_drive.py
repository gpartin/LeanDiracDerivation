"""Two-layer matched-clock frequency decomposition of the native face current.

This is a lossless diagnostic of Psi/W on actual committed layers, not an
extra evolution register. An unforced clock rotates the interference by
twice its rest angle. The actual centered clock defect drives departures
from that beat, giving the exact native current-sustaining requirement.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import unpack,face_charge,divergence,mask_sum,ad_charge

def main():
    ap=argparse.ArgumentParser()
    for x in ("rest-cache","archive","output"):ap.add_argument("--"+x,type=Path,required=True)
    ap.add_argument("--axis",type=int,default=0);ap.add_argument("--sign",type=int,default=-1)
    a=ap.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    ref=np.load(a.rest_cache)["rest"];st=np.load(a.archive)
    dt=float(st["dt"]);states=st["states"];axis=a.axis;sign=a.sign
    p,c=unpack(ref);core=np.linalg.norm(p,axis=-1)>1e-12;receiver=np.roll(core,sign,axis)&~core
    angle=np.arccos(1-dt*dt*rank.R**2/2);cos=np.cos(angle);sin=np.sin(angle);u=np.exp(1j*angle)
    clock=u.conjugate()**2;K=2*rank.B*sin**2/dt**2
    def split(l,r):
        pos=(r-u.conjugate()*l)/(2j*sin)
        neg=(u*l-r)/(2j*sin)
        return pos,neg
    def c_edges(pos,neg):
        out=[]
        for j in range(3):
            pj=np.roll(pos,-1,j);nj=np.roll(neg,-1,j)
            out.append(u.conjugate()*np.sum(pj.conj()*neg-pos.conj()*nj,axis=-1))
        return np.asarray(out)
    def cut(e):
        return np.asarray([np.sum((np.roll(x,1,j)-x)[receiver]) for j,x in enumerate(e)])
    def pair(z):
        return {"real":z.real.tolist(),"imag":z.imag.tolist()}
    ps,cs=zip(*(unpack(q) for q in states));ws=[np.cross(q,p)/rank.R for q,p in zip(cs,ps)]
    channels={"Psi":np.asarray(ps),"W":np.asarray(ws)}
    result={}
    for key,fields in channels.items():
        rows=[];advance=[];cc=[];parts=[]
        for n,(l,r) in enumerate(zip(fields,fields[1:])):
            pos,neg=split(l,r)
            C=c_edges(pos,neg);cc.append(cut(C));parts.append((pos,neg))
            _,original=face_charge(states[n],states[n+1],dt)
            raw=original[key][2]
            observed=cut(raw/dt)
            rec=K*cut(C).imag
            rows.append({"interval":n-1,"positive_coordinate_norm":float(np.linalg.norm(pos)),
                         "negative_coordinate_norm":float(np.linalg.norm(neg)),
                         "reconstruction_error":max(float(np.linalg.norm(l-pos-neg)),float(np.linalg.norm(r-u*pos-u.conjugate()*neg))),
                         "direct_momentum_rate_by_axis":observed.tolist(),
                         "counterrotating_current_by_axis":rec.tolist(),
                         "frequency_current_relative_defect":float(np.linalg.norm(observed-rec)/max(np.linalg.norm(observed),1.))})
        for i in range(len(parts)-1):
            pos,neg=parts[i];pos1,neg1=parts[i+1]
            defect=fields[i]+fields[i+2]-2*cos*fields[i+1]
            T=defect/(2j*sin)
            free=c_edges(u*pos,u.conjugate()*neg)
            actual=c_edges(u*pos+T,u.conjugate()*neg-T)
            D=cut(actual-free)
            exact=cc[i+1]-clock*cc[i]
            # A sustained slowly moving envelope requires the negative
            # clock coordinate to co-rotate with the positive one, rather
            # than freely beat against it. The exact zero-envelope-rate
            # control is R=4 sin²(theta) N. This is not imposed on evolution.
            lock=4*sin**2*neg
            locknorm=max(float(np.linalg.norm(lock)),1e-300)
            advance.append({"centered_layer":i+1,"native_defect_norm":float(np.linalg.norm(defect)),
                            "corotating_N_required_defect_norm":float(np.linalg.norm(lock)),
                            "corotating_N_defect_relative_mismatch":float(np.linalg.norm(defect-lock)/locknorm),
                            "corotating_N_native_defect_fraction":float(np.real(np.vdot(lock,defect))/locknorm**2),
                            "frequency_update_error":max(float(np.linalg.norm(pos1-u*pos-T)),float(np.linalg.norm(neg1-u.conjugate()*neg+T))),
                            "drive_identity_absolute_error":float(np.linalg.norm(D-exact)),
                            "interference_drive":pair(D),
                            "current_rate_change_observed":float(K*(cc[i+1].imag-cc[i].imag).sum()),
                            "current_rate_change_unforced_rotation":float(K*((clock-1)*cc[i]).imag.sum()),
                            "current_rate_change_native_drive":float(K*D.imag.sum())})
        second=[]
        for i in range(1,len(cc)-1):
            D0=cc[i]-clock*cc[i-1];D1=cc[i+1]-clock*cc[i]
            I=np.array([K*z.imag.sum() for z in cc[i-1:i+2]])
            native=float(K*(D1-clock.conjugate()*D0).imag.sum())
            required=float(2*(1-clock.real)*I[1])
            second.append({"centered_interval":i-1,
                           "current_rate":float(I[1]),"observed_second_difference":float(I[2]-2*I[1]+I[0]),
                           "natural_beat_second_difference":float(2*(clock.real-1)*I[1]),
                           "native_drive_second_difference":native,
                           "constant_rate_required_drive":required,
                           "drive_fraction_of_constant_rate_requirement":native/required if abs(required)>1e-30 else None,
                           "identity_error":float(abs(I[2]-2*clock.real*I[1]+I[0]-native))})
        # Actual release starts at the interval labelled 0 (index 1).
        initial=cc[1]
        two_cycles=np.arange(round(4*np.pi/angle)+1)
        free_rates=np.asarray([K*(clock**int(n)*initial).imag.sum() for n in two_cycles])
        result[key]={"intervals":rows,"advances":advance,"second_difference":second,
                     "no_drive_initial_current_rate":float(K*initial.imag.sum()),
                     "no_drive_beat_period_steps":float(np.pi/angle),
                     "no_drive_first_nonpositive_step":int(np.flatnonzero(free_rates<=0)[0]) if np.any(free_rates<=0) else None,
                     "no_drive_two_cycle_average_current_rate":float(np.mean(free_rates))}
    # Total native through-current is independently measured. For Psi the
    # two matched rotating channels have the same beat; the other terms
    # are recorded separately rather than assigned the direct oscillator law.
    combined=[]
    for i in range(len(states)-1):
        l,r=states[i:i+2]
        ql,qr=ad_charge(action.interval,l,r,dt);full=mask_sum(ql+qr,receiver)/dt
        vals=[result[k]["intervals"][i]["direct_momentum_rate_by_axis"] for k in ("Psi","W")]
        matched=float(np.sum(vals))
        combined.append({"interval":i-1,"native_Psi_rate":float(full[0]),"native_chi_rate":float(full[1]),"matched_Psi_rate":matched,"other_Psi_rate":float(full[0]-matched)})
    out={"status":"V4_006_NATIVE_CLOCK_INTERFERENCE_AND_REQUIRED_DRIVE_DIAGNOSTIC",
         "action_sha256":sha256(Path(action.__file__)),"rest_cache_sha256":sha256(a.rest_cache),"state_sha256":sha256(a.archive),
         "dt":dt,"axis":axis,"sign":sign,"theta":float(angle),"clock_beat_real":float(clock.real),"current_prefactor":float(K),
         "channels":result,"full_native_current":combined,
         "claim_boundary":"Exact two-layer decomposition and driven-current recurrence of native matched face channels. The unforced continuation is an algebraic diagnostic, not an alternate native action or physical worldline; the two clock coordinates are not the historical Clifford copies."}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("clock angle",angle,"beat",np.pi/angle)
    for k,v in result.items():
        print(k,"first no-drive nonpositive",v["no_drive_first_nonpositive_step"],"drive fractions",[x["drive_fraction_of_constant_rate_requirement"] for x in v["second_difference"]])
    print("native",[(x["interval"],x["native_Psi_rate"],x["native_chi_rate"]) for x in combined])

if __name__=="__main__":main()
