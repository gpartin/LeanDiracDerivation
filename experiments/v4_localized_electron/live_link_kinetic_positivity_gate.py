"""Analytic positivity bound for the normalized C19 joint kinetic link."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHI0 = 19.0
G = 1.0/CHI0
C19_WEIGHT_SUM = 24.0


def main() -> None:
    # For complex C3 vectors the Hermitian Gram identity gives
    # |chi_i wedge chi_j|^2 = rho_i rho_j-|chi_i^dagger chi_j|^2,
    # hence every normalized gate lies in [0,1].  The absolute row sum of the
    # normalized weighted C19 adjacency is at most one.  Gershgorin therefore
    # bounds both native kinetic metrics by I +/- |g|.
    lower = 1-abs(G)
    upper = 1+abs(G)
    result = {
        "status": "V4_LIVE_LINK_KINETIC_POSITIVITY_GATE",
        "candidate": "joint_c19_wedge_metric_sq",
        "coefficient": G,
        "coefficient_provenance": "1/chi0 with chi0=19",
        "C19_weight_sum": C19_WEIGHT_SUM,
        "normalized_wedge_gate_bounds": [0.0, 1.0],
        "kinetic_metric_eigenvalue_bounds": [lower, upper],
        "strictly_positive_lower_bound": bool(lower > 0),
        "verdict": (
            "JOINT_C19_KINETIC_METRIC_POSITIVE"
            if lower > 0 else "JOINT_C19_KINETIC_METRIC_NOT_PROVED_POSITIVE"),
        "claim_boundary": (
            "Analytic Gershgorin bound for the added Psi and B-weighted chi "
            "velocity blocks. It does not prove nonlinear carrier motion."),
    }
    (HERE/"live_link_kinetic_positivity_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
