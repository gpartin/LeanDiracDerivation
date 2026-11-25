"""Assemble the bounded verdict for the C19 triangle/gyro repair branch."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


BASES = [
    ("direct chi bond gyros",
     "c27_chi_bond_gyro_obstruction_rank.json",
     "CHI_BOND_GYRO_BASIS_INCOMPLETE_FOR_NONLINEAR_OBSTRUCTION"),
    ("cross-field bond gyros",
     "c27_cross_bond_gyro_obstruction_rank.json",
     "CROSS_BOND_GYRO_BASIS_MISSES_NONLINEAR_OBSTRUCTION"),
    ("shared-neighbor chi orientation response",
     "c19_chi_neighbor_orientation_obstruction_rank.json",
     "C19_NEIGHBOR_ORIENTATION_BASIS_MISSES_NONLINEAR_OBSTRUCTION"),
    ("vacuum-mold alignment potential",
     "c19_vacuum_mold_alignment_obstruction.json",
     "VACUUM_MOLD_ALIGNMENT_MISSES_NONLINEAR_OBSTRUCTION"),
    ("parallel-or-orthogonal chi bond potential",
     "c19_bistable_chi_orientation_gate.json",
     "C19_BISTABLE_CHI_ORIENTATION_NO_IMPROVEMENT"),
    ("cube-overlap geometric chi bond potential",
     "c19_geometric_chi_orientation_gate.json",
     "C19_GEOMETRIC_CHI_ORIENTATION_NO_IMPROVEMENT"),
    ("C19 chi neighbor Berry one-form",
     "c19_chi_bond_gyro_spectrum_gate.json",
     "C19_CHI_BOND_GYRO_NO_IMPROVEMENT"),
    ("local chi Berry one-form",
     "c19_chi_phase_space_basis_gate.json",
     "C19_CHI_PHASE_SPACE_BASIS_NO_IMPROVEMENT"),
    ("C19-gradient chi Berry one-form",
     "c19_chi_phase_space_basis_gate.json",
     "C19_CHI_PHASE_SPACE_BASIS_NO_IMPROVEMENT"),
    ("positive C19-gradient chi kinetic metric",
     "c19_chi_phase_space_basis_gate.json",
     "C19_CHI_PHASE_SPACE_BASIS_NO_IMPROVEMENT"),
]


def main() -> None:
    rows = []
    for name, filename, expected in BASES:
        payload = json.loads((HERE / filename).read_text(encoding="utf-8"))
        rows.append({
            "basis": name,
            "evidence": filename,
            "verdict": payload.get("verdict"),
            "expected_verdict": expected,
            "zero_improvement_confirmed": payload.get("verdict") == expected,
        })
    ten_failed = len(rows) == 10 and all(
        row["zero_improvement_confirmed"] for row in rows)
    result = {
        "status": "V4_C19_TRIANGLE_CANDIDATE_BOUNDED_VERDICT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "incumbent_positive_result": {
            "exact_C19_triangle_factorization": True,
            "exact_rest_DEL_RMS": 2.34e-12,
            "fixed_charge_transport_ranks": [3, 3, 3],
            "nonlinear_0p1_percent_three_cycle_shape_residual":
                0.002463679609827518,
        },
        "failed_acceptance_gates": [
            "one-percent perturbation update does not remain on a solved branch",
            "complete physical 100-cycle stability is not established",
            "no freely translating localized carrier has been found",
            "no two-handoff both-sign all-axis movement pass exists",
            "same-state native-current/Dirac-current closure is absent",
        ],
        "ten_consecutive_zero_improvement_bases": rows,
        "failure_rule_satisfied": ten_failed,
        "exact_unresolved_mode": (
            "exterior and interface chi orientation/clock response coupled to "
            "the carrier translation tangent; potential stiffness and the "
            "complete reduced chi-only Berry/kinetic basis split unit-circle "
            "modes instead of producing a stable translating branch"
        ),
        "scoped_obstruction": (
            "The C19 triangle/gyro incumbent cannot be completed by the ten "
            "tested position, bond-gyro, shared-neighbor, chi-only potential, "
            "or reduced chi-only phase-space action bases."
        ),
        "smallest_untested_action_class": (
            "a genuinely joint, field-dependent Psi--chi kinetic connection "
            "whose symplectic curvature acts on the carrier--vacuum interface; "
            "alternatively a freely solved noncoplanar rest branch before any "
            "new action term"
        ),
        "verdict": (
            "ACTION_BASIS_NO_GO" if ten_failed else
            "BOUNDED_FAILURE_AUDIT_INCOMPLETE"
        ),
        "claim_boundary": (
            "Scoped no-go for the enumerated completion bases only. It is not "
            "a no-go for every local reciprocal two-field action and does not "
            "alter canonical V40."
        ),
    }
    output = HERE / "c19_triangle_candidate_bounded_verdict.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
