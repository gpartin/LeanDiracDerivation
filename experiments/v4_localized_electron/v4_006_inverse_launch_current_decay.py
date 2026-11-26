"""Term-by-term current change along the unrestricted inverse-launch layers."""
from pathlib import Path
import argparse,json
from hashlib import sha256 as hashlib_sha256
import numpy as np
import torch
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
from v4_003_fixed_phase_space_readout import sha256
from v4_006_face_phase_momentum import face_charge,divergence,unpack,mask_sum,ad_charge,dotimag

def main():
    p=argparse.ArgumentParser()
    for name in ("rest-cache","archive","launch-report","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();rank.lock_inertia();torch.set_num_threads(2)
    report=json.loads(a.launch_report.read_text());archive=np.load(a.archive);states=archive["states"];dt=float(archive["dt"])
    axis=report["axis"];sign=report["sign"];rest=np.load(a.rest_cache)["rest"]
    core=np.linalg.norm(unpack(rest)[0],axis=-1)>1e-12;rec=np.roll(core,sign,axis)&~core
    rows=[]
    for n,(l,r) in enumerate(zip(states,states[1:]),start=-1):
        total,parts=face_charge(l,r,dt)
        flx=divergence(total[2]);base={}
        for name,part in parts.items():
            flow=divergence(part[2][None])[0]
            face=np.array([np.roll(part[2][j],1,j)-part[2][j] for j in range(3)])
            rate=float(np.sum(flow[rec])/dt)
            base[name]={"rate":rate,"rate_by_axis":(face[:,rec].sum(axis=1)/dt).tolist()}
        # Independent exact midpoint decomposition of the direct Psi term:
        # matched clock potential versus phase carried by the bond velocity.
        pl,_=unpack(l);pr,_=unpack(r);mid=(pl+pr)/2;vel=(pr-pl)/dt
        kinetic_weight=1-dt*dt*rank.R**2/4
        def cut(values):
            return (np.stack([np.roll(v,1,j)-v for j,v in enumerate(values)])[:,rec]).sum(axis=1)
        position=cut([rank.B*rank.R**2*dotimag(np.roll(mid,-1,j),mid) for j in range(3)])
        velocity=cut([-rank.B*kinetic_weight*dotimag(np.roll(vel,-1,j),vel) for j in range(3)])
        base["Psi"]["midpoint_clock_position_rate_by_axis"]=position.tolist()
        base["Psi"]["midpoint_velocity_rate_by_axis"]=velocity.tolist()
        base["Psi"]["midpoint_recombination_absolute_defect"]=float(np.linalg.norm(position+velocity-base["Psi"]["rate_by_axis"]))
        ql,qr=ad_charge(action.interval,l,r,dt);full=mask_sum(ql+qr,rec)/dt
        full_face=mask_sum(flx,rec)/dt
        rows.append({"interval":n,"native_total_receiver_rate":full.tolist(),"matched_face_rate":full_face.tolist(),
                     "other_action_receiver_rate":(full-full_face).tolist(),"face_channels":base})
    # Derive the exact advance of direct Psi current from its two clock
    # defects. The formula has been proved for each real-coordinate
    # complex component in FacePhaseMomentum.lean.
    cc=1-dt*dt*rank.R**2/2
    advance=[]
    for i in range(1,len(states)-1):
        u,v,w=(unpack(x)[0] for x in states[i-1:i+2])
        delta=w-u;clock_defect=w+u-2*cc*v
        native=[];defect=[]
        for j in range(3):
            n=lambda x:np.roll(x,-1,j)
            native.append(rank.B/dt**2*(1-cc**2)*(dotimag(n(v),delta)+dotimag(n(delta),v)))
            defect.append(-rank.B/dt**2*cc/2*(dotimag(n(delta),clock_defect)+dotimag(n(clock_defect),delta)))
        def integrated(array):
            return np.stack([np.roll(x,1,j)-x for j,x in enumerate(array)])[:,rec].sum(axis=1)
        na=integrated(native);de=integrated(defect)
        change=np.asarray(rows[i]["face_channels"]["Psi"]["rate_by_axis"])-rows[i-1]["face_channels"]["Psi"]["rate_by_axis"]
        advance.append({"centered_layer":i,"observed_rate_change_by_axis":change.tolist(),
                        "centered_clock_rate_change_by_axis":na.tolist(),
                        "clock_defect_rate_change_by_axis":de.tolist(),
                        "identity_absolute_defect":float(np.linalg.norm(change-na-de))})
    f=rows[1];l=rows[-1]
    d={key:l["face_channels"][key]["rate"]-f["face_channels"][key]["rate"] for key in f["face_channels"]}
    out={"status":"V4_006_CURRENT_DECAY_CHANNEL_DECOMPOSITION","action_sha256":sha256(Path(action.__file__)),
         "state_sha256":sha256(a.archive),"rest_cache_sha256":sha256(a.rest_cache),"launch_report_sha256":hashlib_sha256(a.launch_report.read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
         "dt":dt,"axis":axis,"sign":sign,"rows":rows,"direct_Psi_current_advance_clock_split":advance,"change_after_three_unrestricted_updates_by_face_channel":d,
         "native_total_change_after_three_unrestricted_updates":(np.array(l["native_total_receiver_rate"])-np.array(f["native_total_receiver_rate"])).tolist(),
         "other_action_change_after_three_unrestricted_updates":(np.array(l["other_action_receiver_rate"])-np.array(f["other_action_receiver_rate"])).tolist(),
         "claim_boundary":"Exact local native phase current of the already solved initial and free layers, not a full handoff."}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
