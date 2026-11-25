"""Unrestricted nonlinear short-run gate for the extended packet.

The native nonlinear DEL solver receives only the two initial layers.  The
Dirac packet is evolved independently and is used solely after each update as
a comparator.  No projection, reset, background maintenance, or target force
is applied to the native state.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as gate
import cube_incidence_implicit_update as update
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent
UPDATES = 4
AMPLITUDE = 1e-4


def complex_field(real_coordinates):
    return (real_coordinates.reshape(-1, 2)
            @np.asarray((1.0, 1.0j))).reshape(gate.SIDE, gate.SIDE,
                                               gate.SIDE, 4)


def main():
    bare, matrices = action.expected_blocks()
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    factor = lu_factor(update.rest_following_jacobian(rotation, matrices))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    # Axis and diagonal, both signs, in both exhaustive Clifford copies.
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        qbasis = envelope.orth(raw)
        for axis, sign in itertools.product((0, 3), (-1, 1)):
            packet_previous, packet_current, _packet_following, momentum = (
                gate.packet(axis, sign, alpha, beta))
            q_previous = raw@gate.real_coordinates(packet_previous)
            q_current = raw@gate.real_coordinates(packet_current)
            scale = AMPLITUDE*np.linalg.norm(bare["rest"])/np.linalg.norm(q_current)
            q_previous *= scale
            q_current *= scale
            rest_previous = rotation.T@bare["rest"]
            rest_current = bare["rest"].copy()
            previous = rest_previous+rotation.T@q_previous
            current = rest_current+q_current
            dirac_current = packet_current.copy()
            rest_layer = rest_current.copy()
            unrotation_count = 0
            samples = []
            converged = True
            for step in range(1, UPDATES+1):
                dirac_current = gate.dirac_step(
                    dirac_current, alpha, beta, 1, native_time.DT)
                rest_next = rotation@rest_layer
                guess = 2*current-previous
                following, history, passed = update.solve_following(
                    previous, current, guess, factor,
                    tolerance=1e-10, maximum_iterations=20)
                if not passed:
                    converged = False
                    samples.append({
                        "step": step,
                        "solver_converged": False,
                        "solver_residual_RMS": history[-1],
                    })
                    break
                perturbation = following-rest_next
                unrotation_count += 1
                corotating = perturbation.copy()
                for _ in range(unrotation_count):
                    corotating = rotation.T@corotating
                recovered, *_ = np.linalg.lstsq(raw, corotating, rcond=1e-10)
                reconstructed = raw@recovered
                native_field = complex_field(recovered)/scale
                target_raw = scale*raw@gate.real_coordinates(dirac_current)
                samples.append({
                    "step": step,
                    "solver_converged": True,
                    "newton_iterations": len(history)-1,
                    "native_state_vs_dirac_lift_relative_error": float(
                        np.linalg.norm(corotating-target_raw)
                        / np.linalg.norm(target_raw)),
                    "ill_conditioned_coordinate_readout_error": float(
                        np.linalg.norm(native_field-dirac_current)
                        / np.linalg.norm(dirac_current)),
                    "outside_packet_lift_fraction": float(
                        np.linalg.norm(corotating-reconstructed)
                        / max(np.linalg.norm(corotating), 1e-30)),
                    "native_packet_norm_ratio": float(
                        np.linalg.norm(native_field)/np.linalg.norm(dirac_current)),
                })
                previous, current = current, following
                rest_layer = rest_next
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "x",
                "sign": sign,
                "momentum": momentum.tolist(),
                "completed_updates": len(samples),
                "converged": converged,
                "samples": samples,
            })
    physical = [sample for row in rows for sample in row["samples"]
                if sample.get("solver_converged")]
    all_converged = all(row["converged"]
                        and row["completed_updates"] == UPDATES for row in rows)
    worst_error = max((sample["native_state_vs_dirac_lift_relative_error"]
                       for sample in physical),
                      default=float("inf"))
    worst_leakage = max((sample["outside_packet_lift_fraction"]
                         for sample in physical), default=float("inf"))
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_NONLINEAR",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "amplitude_fraction_of_rest_norm": AMPLITUDE,
        "requested_updates": UPDATES,
        "rows": rows,
        "all_solvers_converged": all_converged,
        "maximum_relative_packet_error": worst_error,
        "maximum_outside_packet_lift_fraction": worst_leakage,
        "verdict": ("EXTENDED_PACKET_NONLINEAR_SHORT_RUN_PASS"
                    if all_converged and worst_error < 3e-2
                    and worst_leakage < 2e-2
                    else "EXTENDED_PACKET_NONLINEAR_OPEN"),
        "claim_boundary": (
            "Four unrestricted nonlinear updates at 0.01% tangent amplitude "
            "for both copies, both signs, one axis and a diagonal. This is a "
            "short-run validation, not two cell translations, 100 cycles, "
            "box refinement, or a final V4-006 pass."),
    }
    (HERE/"cube_incidence_extended_packet_nonlinear.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
