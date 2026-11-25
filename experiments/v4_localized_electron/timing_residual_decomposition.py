"""Split the first native--Dirac mismatch into time-even and time-odd parts.

For the co-rotating tangent DEL polynomial

    A q_(n+1) + B q_n + C q_(n-1) = 0,

the residual on the independently generated centered-Dirac layers is

    r_even = B q_n + (A+C)(q_(n+1)+q_(n-1))/2,
    r_odd  =       (A-C)(q_(n+1)-q_(n-1))/2.

The first term is the stiffness/frequency-curvature channel.  The second is
the velocity/symplectic channel.  This is a diagnostic decomposition only;
neither term is installed as a force.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
import live_clifford_tangent_gate as tangent


HERE = Path(__file__).resolve().parent


def relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value)/max(np.linalg.norm(reference), 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--wave-number", type=float, default=0.35)
    parser.add_argument("--packet-width", type=float, default=1.35)
    args = parser.parse_args()

    scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
    plaquette_tag = format(
        action.PLAQUETTE_SCALE, ".12g").replace(".", "p").replace("-", "m")
    spacetime_tag = format(
        action.SPACETIME_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    symplectic_tag = format(
        action.SYMPLECTIC_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_tag = format(
        action.PARALLEL_OVERLAP_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_face_tag = format(
        action.PARALLEL_FACE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_matched_tag = format(
        action.PARALLEL_MATCHED_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_potential_tag = format(
        action.PARALLEL_MATCHED_POTENTIAL_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    dt_tag = format(tangent.DT, ".12g").replace(".", "p").replace("-", "m")
    tangent.CACHE = HERE / (
        f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}"
        f"_plaquette_{plaquette_tag}_spacetime_{spacetime_tag}"
        f"_symplectic_{symplectic_tag}_overlap_{overlap_tag}"
        f"_overlapface_{overlap_face_tag}"
        f"_overlapmatched_{overlap_matched_tag}"
        f"_overlappotential_{overlap_potential_tag}_dt_{dt_tag}.npz")

    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-tangent.DT**2*rank.R**2/2))
    rotate_minus = tangent.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = tangent.phase_rotation_matrix(rest.size, theta)
    blocks = tangent.tangent_blocks(
        (rotate_minus@rest, rest, rotate_plus@rest),
        rebuild=args.rebuild_cache)
    previous, current, following = blocks
    a = following@rotate_plus
    b = current
    c = previous@rotate_minus
    factor = splu(tangent.sparse(a))

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    rows = []
    for (flavor, basis), axis, sign in itertools.product(
            dirac.flavor_bases().items(), (0, 3), (-1, 1)):
        alpha, beta = dirac.compressed_clifford(basis)
        raw = tangent.packet_columns(basis, psi_frame, chi_frame, sites, 0j)
        retained = envelope.orth(raw)
        p, q, f, momentum = dirac.packet(
            axis, sign, alpha, beta, tangent.DT,
            width=args.packet_width,
            carrier_wave_number=args.wave_number,
            centered_time=True,
            frequency_branch="negative")
        qp = raw@dirac.real_coordinates(p)
        qq = raw@dirac.real_coordinates(q)
        qf = raw@dirac.real_coordinates(f)

        r_even = b@qq + (a+c)@(qf+qp)/2
        r_odd = (a-c)@(qf-qp)/2
        r_total = r_even+r_odd
        delta_even = -factor.solve(r_even)
        delta_odd = -factor.solve(r_odd)
        delta_total = delta_even+delta_odd
        outside_even = r_even-retained@(retained.T@r_even)
        outside_odd = r_odd-retained@(retained.T@r_odd)
        denominator = max(np.linalg.norm(r_even)*np.linalg.norm(r_odd), 1e-30)
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "even_residual_norm": float(np.linalg.norm(r_even)),
            "odd_residual_norm": float(np.linalg.norm(r_odd)),
            "total_residual_norm": float(np.linalg.norm(r_total)),
            "even_to_total_norm_ratio": float(
                np.linalg.norm(r_even)/max(np.linalg.norm(r_total), 1e-30)),
            "odd_to_total_norm_ratio": float(
                np.linalg.norm(r_odd)/max(np.linalg.norm(r_total), 1e-30)),
            "even_odd_cosine": float(np.dot(r_even, r_odd)/denominator),
            "even_first_step_state_error": relative(delta_even, qf),
            "odd_first_step_state_error": relative(delta_odd, qf),
            "total_first_step_state_error": relative(delta_total, qf),
            "even_outside_retained_fraction": relative(outside_even, r_even),
            "odd_outside_retained_fraction": relative(outside_odd, r_odd),
        })

    even_errors = np.asarray([row["even_first_step_state_error"] for row in rows])
    odd_errors = np.asarray([row["odd_first_step_state_error"] for row in rows])
    result = {
        "status": "V4_006_TIMING_RESIDUAL_DECOMPOSITION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "plaquette_scale": action.PLAQUETTE_SCALE,
        "spacetime_plaquette_scale": action.SPACETIME_PLAQUETTE_SCALE,
        "symplectic_plaquette_scale": action.SYMPLECTIC_PLAQUETTE_SCALE,
        "parallel_overlap_scale": action.PARALLEL_OVERLAP_SCALE,
        "parallel_face_scale": action.PARALLEL_FACE_SCALE,
        "parallel_matched_scale": action.PARALLEL_MATCHED_SCALE,
        "parallel_matched_potential_scale": (
            action.PARALLEL_MATCHED_POTENTIAL_SCALE),
        "frequency_branch": "negative",
        "dirac_time": "centered",
        "rows": rows,
        "maximum_even_first_step_state_error": float(np.max(even_errors)),
        "maximum_odd_first_step_state_error": float(np.max(odd_errors)),
        "mean_even_first_step_state_error": float(np.mean(even_errors)),
        "mean_odd_first_step_state_error": float(np.mean(odd_errors)),
        "dominant_channel": (
            "TIME_EVEN_STIFFNESS" if np.mean(even_errors) > np.mean(odd_errors)
            else "TIME_ODD_SYMPLECTIC"),
        "claim_boundary": (
            "Exact algebraic split of the first centered-Dirac DEL residual "
            "at the localized rest tangent. It is not an action repair or "
            "nonlinear movement pass."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
