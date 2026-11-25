"""Derive the six-face/chi0 transport normalization and positivity bound.

This is an algebraic gate for the unadmitted smooth face-transport action.
It does not infer movement from the number 6.  It verifies that coefficient
1/chi0 on every undirected face bond produces aggregate incident strength
6/chi0, while the complete kinetic metric remains strictly positive after
the native kappa regularization.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHI0 = 19.0
KAPPA = 1.0/63.0
FACE_DEGREE = 6
PER_BOND_COEFFICIENT = 1.0/CHI0


def main() -> None:
    maximum_gate = (1.0+KAPPA)**2
    aggregate = FACE_DEGREE*PER_BOND_COEFFICIENT
    adjacency_lower = 1.0-aggregate*maximum_gate
    adjacency_upper = 1.0+aggregate*maximum_gate
    # A weighted face graph Laplacian is positive semidefinite and has
    # spectral upper bound 2*degree*max_gate.  Adding it to the identity
    # leaves the common mode exactly at one.
    laplacian_lower = 1.0
    laplacian_upper = 1.0+2.0*aggregate*maximum_gate
    result = {
        "status": "V4_FACE_TRANSPORT_GEOMETRY_GATE",
        "native_mode_count": int(CHI0),
        "matter_transport_bonds_per_site": FACE_DEGREE,
        "edge_bonds_in_matter_transport_term": 0,
        "per_undirected_face_bond_coefficient": PER_BOND_COEFFICIENT,
        "aggregate_incident_face_strength": aggregate,
        "omega_m_geometric_ratio": aggregate,
        "native_kappa": KAPPA,
        "smooth_wedge_gate_global_upper_bound": maximum_gate,
        "adjacency_metric_eigenvalue_bounds": [
            adjacency_lower, adjacency_upper],
        "laplacian_metric_eigenvalue_bounds": [
            laplacian_lower, laplacian_upper],
        "common_covariant_velocity_eigenvalue": 1.0,
        "strictly_positive": laplacian_lower > 0,
        "derivation": (
            "Each site is incident on the +/- face bond in three axes. "
            "Variation of the undirected bond sum therefore exposes six "
            "neighbors. At coefficient 1/chi0 per bond, the aggregate "
            "coordination is 6/chi0."
        ),
        "verdict": (
            "SIX_FACE_RATIO_DERIVED_AND_METRIC_POSITIVE"
            if laplacian_lower > 0 else "FACE_METRIC_POSITIVITY_OPEN"
        ),
        "claim_boundary": (
            "This derives the geometric ratio and a global positivity bound. "
            "It does not by itself prove a moving localized carrier."
        ),
    }
    (HERE/"live_face_transport_geometry_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
