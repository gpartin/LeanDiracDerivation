"""Hash and verify the current-advance research reports (integrity only)."""
from pathlib import Path
from hashlib import sha256
import json,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def digest(p):
    b=p.read_bytes()
    if p.suffix in (".json",".py",".md",".lean"):b=b.replace(b"\r\n",b"\n")
    return sha256(b).hexdigest()

def finite(v):
    if isinstance(v,float) and not math.isfinite(v):raise ValueError("non-finite metric")
    if isinstance(v,list):
        for x in v:finite(x)
    if isinstance(v,dict):
        for x in v.values():finite(x)

def main():
    scripts={
        "current_advance_inverse":["current_advance_inverse","current_advance_inverse_matched_energy"],
        "current_curvature_inverse":["current_curvature_screen"],
        "clock_interference_drive":["clock_interference_drive_old","clock_interference_drive_advance","clock_interference_drive_matched_energy"],
        "native_clock_drive_attribution":["native_clock_drive_attribution"],
        "prong_clock_drive_response":["prong_clock_drive_response"]}
    states=["antiphase_handoff","current_inverse_launch","current_advance_inverse","current_advance_inverse_matched_energy"]
    states={digest(HERE/f"v4_006_{n}.npz"):f"v4_006_{n}.npz" for n in states}
    action=digest(HERE/"live_clifford_bond_action.py");rest=digest(HERE/"v4_moving_rest_seed.npz")
    sources={f"v4_006_{n}.py":digest(HERE/f"v4_006_{n}.py") for n in scripts}
    reports={}
    for source,names in scripts.items():
        for n in names:
            p=HERE/f"v4_006_{n}.json";d=json.loads(p.read_text());finite(d)
            if d["action_sha256"]!=action or d["rest_cache_sha256"]!=rest:raise ValueError(f"wrong action/rest state {p}")
            for key in ("state_sha256","warm_start_sha256","output_state_sha256"):
                if key in d and d[key] not in states:raise ValueError(f"unknown {key} in {p}")
            reports[p.name]={"sha256":digest(p),"source":f"v4_006_{source}.py","status":d["status"]}
    tracker=json.loads((HERE/"v4_coherent_stage_tracker.json").read_text())
    gates={x["id"]:x["status"] for x in tracker["stages"]}
    if any(gates[f"V4-{i:03}"]!="PASS" for i in range(1,6)) or gates["V4-006"]!="OPEN":raise ValueError("incorrect V4 scope")
    proofs={f"LeanDirac/V4/{n}.lean":digest(ROOT/f"LeanDirac/V4/{n}.lean") for n in ("CurrentAdvanceSensitivity","FacePhaseMomentum")}
    result={"status":"RESEARCH_EVIDENCE_INTEGRITY_PASS__NOT_MOVEMENT_PASS","action_sha256":action,"rest_cache_sha256":rest,
            "state_artifacts":states,"source_hashes":sources,"research_Lean_sources":proofs,"reports":reports,"stage_status":gates,
            "claim_boundary":"Checks finite evidence, action and state provenance, and status scope. Lean proof and physical motion are separate checks."}
    (HERE/"v4_006_current_advance_artifact_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Verified {len(reports)} current-advance research reports and {len(states)} state artifacts; V4-006 OPEN.")

if __name__=="__main__":main()
