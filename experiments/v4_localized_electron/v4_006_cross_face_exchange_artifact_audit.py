"""Check the cross-face research evidence's hashes and scope (not a gate)."""
from pathlib import Path
import json, math
from hashlib import sha256
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def digest(p):
    data=p.read_bytes()
    if p.suffix in (".json",".md",".py",".lean"):data=data.replace(b"\r\n",b"\n")
    return sha256(data).hexdigest()

def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError("non-finite metric")
    if isinstance(x,dict):
        for v in x.values():finite(v)
    if isinstance(x,list):
        for v in x:finite(v)

def main():
    scripts={"oriented_phase_exchange":["oriented_phase_exchange_antiphase","oriented_phase_exchange_matched","oriented_phase_exchange_balanced"],
             "parallel_face_charge_completion":["parallel_face_charge_completion"],
             "phase_sum_difference":["phase_sum_difference"],
             "current_inverse_launch":["current_inverse_launch","current_inverse_launch_energy"],
             "inverse_launch_current_decay":["inverse_launch_current_decay"]}
    states=["antiphase_handoff","antiphase_phase_balanced_handoff","face_capacity_matched_speed","current_inverse_launch","current_inverse_launch_energy"]
    state_hashes={digest(HERE/f"v4_006_{s}.npz"):f"v4_006_{s}.npz" for s in states}
    sources={f"v4_006_{s}.py":digest(HERE/f"v4_006_{s}.py") for s in scripts}
    action_hash=digest(HERE/"live_clifford_bond_action.py")
    rest_hash=digest(HERE/"v4_moving_rest_seed.npz")
    reports={}
    for source,outputs in scripts.items():
        for s in outputs:
            p=HERE/f"v4_006_{s}.json";d=json.loads(p.read_text());finite(d)
            if d["action_sha256"]!=action_hash or d["rest_cache_sha256"]!=rest_hash:raise ValueError(f"wrong incumbent or rest source: {p}")
            for key in ("state_sha256","output_state_sha256"):
                if key in d and d[key] not in state_hashes:raise ValueError(f"missing state {key}: {p}")
            if "launch_report_sha256" in d and d["launch_report_sha256"]!=digest(HERE/"v4_006_current_inverse_launch.json"):raise ValueError("wrong launch report")
            reports[p.name]={"sha256":digest(p),"source":f"v4_006_{source}.py","status":d["status"]}
    stages=json.loads((HERE/"v4_coherent_stage_tracker.json").read_text())["stages"]
    scopes={x["id"]:x["status"] for x in stages}
    if any(scopes[f"V4-{i:03}"]!="PASS" for i in range(1,6)) or scopes["V4-006"]!="OPEN":raise ValueError("incorrect stage scope")
    proof=ROOT/"LeanDirac/V4/FacePhaseMomentum.lean"
    result={"status":"RESEARCH_EVIDENCE_INTEGRITY_PASS__NOT_MOVEMENT_PASS","action_sha256":action_hash,"rest_cache_sha256":rest_hash,
            "sources":sources,"state_artifacts":state_hashes,"reports":reports,"stage_status":scopes,
            "research_lean_source":{"path":str(proof.relative_to(ROOT).as_posix()),"sha256":digest(proof)},
            "claim_boundary":"Integrity and scope only; independently compile the Lean research proof and chain."}
    (HERE/"v4_006_cross_face_exchange_artifact_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Verified {len(reports)} cross-face research reports and {len(states)} state artifacts; V4-006 OPEN.")
if __name__=="__main__":main()
