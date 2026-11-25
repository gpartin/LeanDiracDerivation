"""Extract the native linear response from opposite nonlinear amplitudes.

The odd combination cancels the rest orbit and every even nonlinear response.
It is therefore an independent finite-difference check of the cached action
tangent and of the selected discrete-Dirac frequency branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frequency-branch",
                        choices=("positive", "negative"),
                        default="negative")
    args = parser.parse_args()

    positive = np.load(HERE/args.positive)
    negative = np.load(HERE/args.negative)
    plus_states = positive["states"]
    minus_states = negative["states"]
    rest = positive["rest"]
    rotation = positive["rotation"]

    bare, _ = prior_action.expected_blocks()
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    basis = dirac.flavor_bases()["plus"]
    alpha, beta = dirac.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, dirac.OFFSETS, 0j)
    qbasis = envelope.orth(raw)
    _previous, target, _following, momentum = dirac.packet(
        0, -1, alpha, beta, native_time.DT, centered_time=True,
        frequency_branch=args.frequency_branch)

    # State index zero is the previous layer and index one is the current
    # layer.  The requested updates begin at index two.
    odd_current = (plus_states[1]-minus_states[1])/2
    scale = np.linalg.norm(odd_current) / np.linalg.norm(
        raw@dirac.real_coordinates(target))
    rows = []
    dirac_current = target.copy()
    for state_index in range(1, len(plus_states)):
        step = state_index-1
        if step > 0:
            dirac_current = dirac.centered_dirac_step(
                dirac_current, alpha, beta, 1, native_time.DT,
                args.frequency_branch)
        odd = (plus_states[state_index]-minus_states[state_index])/2
        corotating = np.linalg.matrix_power(rotation.T, step)@odd
        expected = scale*raw@dirac.real_coordinates(dirac_current)
        retained = qbasis@(qbasis.T@corotating)
        rows.append({
            "step": step,
            "native_vs_dirac_relative_error": float(
                np.linalg.norm(corotating-expected)/np.linalg.norm(expected)),
            "outside_four_component_fraction": float(
                np.linalg.norm(corotating-retained)
                / max(np.linalg.norm(corotating), 1e-30)),
            "native_norm_ratio": float(
                np.linalg.norm(corotating)/np.linalg.norm(expected)),
        })
    result = {
        "status": "V4_LIVE_FACE_CENTRAL_TANGENT_GATE",
        "frequency_branch": args.frequency_branch,
        "momentum": momentum.tolist(),
        "rows": rows,
        "maximum_native_vs_dirac_relative_error": max(
            row["native_vs_dirac_relative_error"] for row in rows),
        "maximum_outside_four_component_fraction": max(
            row["outside_four_component_fraction"] for row in rows),
        "claim_boundary": (
            "Odd finite-amplitude response for one copy and one direction; "
            "it is a tangent validation, not a V4-006 pass."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
