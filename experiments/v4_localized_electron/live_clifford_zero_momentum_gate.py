"""Separate the rest Dirac grading from the live-link transport response."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"
PSI_LINK = HERE / "live_clifford_tangent_blocks_unit_link_psi_frame_phase.npz"


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def evaluate(path, rest, rotate_minus, rotate_plus, psi_frame, chi_frame, sites):
    data = np.load(path)
    previous_block, current_block, following_block = tuple(
        data[name] for name in ("previous", "current", "following"))
    factor = splu(sparse(following_block@rotate_plus))
    rows = []
    psi_mask = np.zeros(rest.size, dtype=bool)
    for site_index in range(rest.size//12):
        psi_mask[site_index*12:site_index*12+6] = True
    for flavor, basis in dirac.flavor_bases().items():
        _alpha, beta = dirac.compressed_clifford(basis)
        raw = gate.psi_columns(basis, psi_frame, chi_frame, sites)
        hamiltonian = rank.R*(beta-np.eye(4))
        forward = expm(-1j*native_time.DT*hamiltonian)
        backward = expm(1j*native_time.DT*hamiltonian)
        for component in range(4):
            field = np.zeros((5, 5, 5, 4), dtype=complex)
            field[..., component] = 1
            previous = np.einsum("ab,...b->...a", backward, field)
            following = np.einsum("ab,...b->...a", forward, field)
            q_previous = raw@dirac.real_coordinates(previous)
            q_current = raw@dirac.real_coordinates(field)
            q_target = raw@dirac.real_coordinates(following)
            q_native = factor.solve(-(
                sparse(current_block)@q_current
                + sparse(previous_block@rotate_minus)@q_previous))
            rows.append({
                "flavor": flavor,
                "component": component,
                "psi_relative_error": float(
                    np.linalg.norm((q_native-q_target)[psi_mask])
                    / np.linalg.norm(q_target[psi_mask])),
                "induced_chi_fraction": float(
                    np.linalg.norm(q_native[~psi_mask])
                    / max(np.linalg.norm(q_native), 1e-30)),
            })
    return {
        "maximum_psi_relative_error": max(row["psi_relative_error"] for row in rows),
        "maximum_induced_chi_fraction": max(row["induced_chi_fraction"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    incumbent = evaluate(
        INCUMBENT, rest, rotate_minus, rotate_plus, psi_frame, chi_frame, sites)
    live_link = evaluate(
        PSI_LINK, rest, rotate_minus, rotate_plus, psi_frame, chi_frame, sites)
    result = {
        "status": "V4_LIVE_CLIFFORD_ZERO_MOMENTUM_GATE",
        "incumbent": incumbent,
        "psi_live_link": live_link,
        "verdict": (
            "LIVE_LINK_PRESERVES_ZERO_MOMENTUM_DIRAC_GRADING"
            if live_link["maximum_psi_relative_error"] < 1e-3 else
            "ZERO_MOMENTUM_DIRAC_GRADING_REMAINS_OPEN"),
        "claim_boundary": (
            "One-step full two-field tangent at zero spatial momentum. "
            "This separates mass/time grading from spatial transport."),
    }
    (HERE/"live_clifford_zero_momentum_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "incumbent_maximum_error": incumbent["maximum_psi_relative_error"],
        "live_link_maximum_error": live_link["maximum_psi_relative_error"],
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
