"""Rest-force gate for the live-Clifford bond action candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as native_action
import live_clifford_bond_action as candidate


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare, _ = native_action.expected_blocks()
    rest = bare["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    residual = candidate.del_residual(*layers, dt).detach().numpy()
    result = {
        "status": "V4_LIVE_CLIFFORD_BOND_REST_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "scale": candidate.SCALE,
        "chi_payload": candidate.CHI_PAYLOAD,
        "link_source": candidate.LINK_SOURCE,
        "rest_DEL_norm": float(np.linalg.norm(residual)),
        "rest_DEL_rms": float(np.linalg.norm(residual)/np.sqrt(rest.size)),
        "verdict": (
            "LIVE_CLIFFORD_BOND_REST_NULL"
            if np.linalg.norm(residual)/np.sqrt(rest.size) < 1e-10 else
            "LIVE_CLIFFORD_BOND_DEFORMS_REST"),
        "claim_boundary": (
            "Exact retained three-layer rest orbit only. No tangent closure, "
            "spectrum, movement, or nonlinear claim."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    output = HERE/f"live_clifford_bond_rest_gate_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
