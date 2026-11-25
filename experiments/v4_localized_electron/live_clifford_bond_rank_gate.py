"""Rank and cross-block gate for the live-Clifford action increment.

The tested carrier bases are eigenspaces of the orientation involution derived
from the live Psi edge phases.  They are not the historical fixed copy labels.
For each basis this file applies the exact current-layer Hessian of only the
new reciprocal action term and measures response outside that same native
rank-four carrier lift.  No Dirac comparator is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as native_action
import current_action_localized_envelope_closure as envelope
import live_clifford_bond_action as candidate
import live_clifford_link_geometry as geometry


HERE = Path(__file__).resolve().parent


def orth(columns: np.ndarray) -> np.ndarray:
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    tolerance = 1e-11*max(float(singular[0]), 1.0)
    return u[:, singular > tolerance]


def main() -> None:
    bare, _ = native_action.expected_blocks()
    rest = bare["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    source_field = "chi" if candidate.LINK_SOURCE.startswith("chi_") else "psi"
    rays = geometry.occupied_rays(rest, source_field)
    generators = [geometry.live_generator(rays, axis) for axis in range(3)]
    flavor = 1j*generators[0]@generators[1]@generators[2]
    identity = np.eye(8, dtype=complex)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)

    def delta_residual(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
        return candidate.del_residual(previous, current, following, dt) \
            - candidate.incumbent.del_residual(
                previous, current, following, dt)

    rows = []
    for name, projector in (("live_plus", (identity+flavor)/2),
                            ("live_minus", (identity-flavor)/2)):
        values, vectors = np.linalg.eigh(
            (projector+projector.conj().T)/2)
        basis = vectors[:, values > 0.5]
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, [(0, 0, 0)],
            1/np.sqrt(native_time.B)+0j)
        q = orth(raw)
        responses = []
        for column in range(q.shape[1]):
            tangent = q[:, column]
            directions = tuple(torch.tensor(value, dtype=torch.float64)
                               for value in (rotation.T@tangent, tangent,
                                             rotation@tangent))
            _value, image = jvp(
                delta_residual, layers, directions)
            responses.append(image.detach().numpy())
        response = np.column_stack(responses)
        leakage = response-q@(q.T@response)
        singular = np.linalg.svd(response, compute_uv=False)
        rows.append({
            "sector": name,
            "complex_dimension": int(basis.shape[1]),
            "real_lift_rank": int(q.shape[1]),
            "response_rank": int(np.sum(
                singular > max(float(singular[0]), 1.0)*1e-10)),
            "relative_cross_block": float(
                np.linalg.norm(leakage)/max(np.linalg.norm(response), 1e-30)),
            "response_norm": float(np.linalg.norm(response)),
            "leakage_norm": float(np.linalg.norm(leakage)),
        })
    passed = all(row["relative_cross_block"] < 1e-8 for row in rows)
    result = {
        "status": "V4_LIVE_CLIFFORD_BOND_RANK_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": candidate.LINK_SOURCE,
        "sector_link_field": source_field,
        "sector_source": (
            "eigenspaces of i times the product of the three live-Psi "
            "edge-overlap generators"),
        "rows": rows,
        "verdict": (
            "LIVE_CLIFFORD_ACTION_INCREMENT_PRESERVES_BOTH_RANK_FOUR_SECTORS"
            if passed else "LIVE_CLIFFORD_ACTION_INCREMENT_LEAKS_SECTORS"),
        "claim_boundary": (
            "Co-rotating three-layer Hessian of the new action increment on "
            "one carrier center. A pass still requires a moving time graph, the "
            "surrounding shell, full spectrum, and nonlinear motion."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    output = HERE/f"live_clifford_bond_rank_gate_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
