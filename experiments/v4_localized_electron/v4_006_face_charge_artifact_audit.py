"""Reconcile V4-006 phase-charge research evidence (not a promotion gate)."""
from pathlib import Path
import json
from hashlib import sha256

HERE=Path(__file__).resolve().parent


def hashfile(path):
    data=path.read_bytes()
    # Git tracks these text artifacts with LF; normalize generated Windows
    # JSON before recording its portable digest.
    if path.suffix in ('.json','.py','.md'):data=data.replace(b'\r\n',b'\n')
    return sha256(data).hexdigest()


def main():
    names={
        'charge_clock_profile':('charge_clock_profile',),
        'phase_corrected_handoff':('phase_corrected_handoff_5','face_capacity_handoff','face_capacity_matched_speed','face_capacity_prong_phase'),
        'charge_reduced_handoff':('charge_reduced_handoff','charge_reduced_face_seed','charge_reduced_optimized'),
        'face_charge_transport_seed':('face_charge_transport_seed',),
        'midpoint_charge_energy':('midpoint_charge_energy',),
        'handoff_force_decomposition':('handoff_force_decomposition',),
        'clock_covariant_face_probe':('clock_covariant_face_probe_10cycles',),
        'material_face_probe':('material_face_probe','material_face_covariant_probe'),
        'integrate_native_clock':('integrated_native_clock',),
        'charge_preserving_receiver_seed':('charge_preserving_receiver_seed',),
        'receiver_seed_release':('receiver_seed_release','receiver_seed_release_newton'),
    }
    action_hash=hashfile(HERE/'live_clifford_bond_action.py');items=[]
    def no_constant(value):raise ValueError(f'nonfinite JSON number: {value}')
    for script,outputs in names.items():
        source=HERE/f'v4_006_{script}.py'
        for output in outputs:
            f=HERE/f'v4_006_{output}.json';data=json.loads(f.read_text(),parse_constant=no_constant)
            actual=data.get('action_sha256',data.get('incumbent_action_sha256'))
            if actual!=action_hash:raise ValueError(f'action provenance mismatch: {f.name}')
            items.append({'script':source.name,'script_sha256':hashfile(source),'artifact':f.name,'artifact_sha256':hashfile(f),'status':data['status']})
    candidates=['clock_covariant_face_action','material_face_action']
    for entry,candidate in [(items[[x['artifact'] for x in items].index('v4_006_clock_covariant_face_probe_10cycles.json')],candidates[0]),
                            (items[[x['artifact'] for x in items].index('v4_006_material_face_probe.json')],candidates[1]),
                            (items[[x['artifact'] for x in items].index('v4_006_material_face_covariant_probe.json')],candidates[1])]:
        path=HERE/f'v4_006_{candidate}.py';data=json.loads((HERE/entry['artifact']).read_text())
        if data['candidate_sha256']!=hashfile(path):raise ValueError(f'candidate hash mismatch: {path.name}')
    states=[HERE/f'v4_006_{name}.npz' for name in ('face_charge_transport_seed','phase_corrected_handoff_5','face_capacity_handoff','face_capacity_matched_speed','face_capacity_prong_phase')]
    by_hash={hashfile(x):x.name for x in states}
    for item in items:
        state_hash=json.loads((HERE/item['artifact']).read_text()).get('state_sha256')
        if state_hash is not None and state_hash not in by_hash:raise ValueError(f'missing state for {item["artifact"]}')
    tracker=json.loads((HERE/'v4_coherent_stage_tracker.json').read_text());gates={r['id']:r['status'] for r in tracker['stages']}
    if any(gates[f'V4-{i:03d}']!='PASS' for i in range(1,6)) or gates['V4-006']!='OPEN':raise ValueError('incorrect scoped stage status')
    out={'status':'RESEARCH_EVIDENCE_INTEGRITY_PASS__NOT_MOVEMENT_PASS','action_sha256':action_hash,'items':items,
         'candidate_sources':{n:hashfile(HERE/f'v4_006_{n}.py') for n in candidates},'state_artifacts':by_hash,'stage_status':gates,
         'claim_boundary':'Checks existence, JSON finiteness, action/candidate/state provenance and scoped tracker only; no physical or Lean gate is closed by this audit.'}
    (HERE/'v4_006_face_charge_artifact_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Checked {len(items)} reports and {len(states)} state artifacts; V4-006 remains OPEN")


if __name__=='__main__':main()
