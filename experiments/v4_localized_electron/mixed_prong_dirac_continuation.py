"""Amplitude continuation for difficult mixed-prong native update roots."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import mixed_prong_action as action
import mixed_prong_dirac_nonlinear as nonlinear


HERE = Path(__file__).resolve().parent
DT = native_time.DT
FRACTIONS = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6,
             1e-5, 3e-5, 1e-4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    args = parser.parse_args()
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_layers = (
        nonlinear.phase_rotate(rest, -theta), rest,
        nonlinear.phase_rotate(rest, theta))
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in rest_layers)
    jacobian = torch.func.jacrev(
        lambda value: action.del_residual(
            tensors[0], tensors[1], value, DT))(tensors[2]).detach().numpy()
    factor = lu_factor(jacobian)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    scale_map = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scale_map[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))
    (flavor, basis), axis, sign = specifications[args.case_index]
    alpha, beta = dirac.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites,
        dirac.OFFSETS, scale_map[flavor])
    qbasis = envelope.orth(raw)
    packet_previous, packet_current, packet_following, momentum = dirac.packet(
        axis, sign, alpha, beta, DT)
    q_previous = raw@dirac.real_coordinates(packet_previous)
    q_current = raw@dirac.real_coordinates(packet_current)
    q_target = raw@dirac.real_coordinates(packet_following)
    unit_scale = np.linalg.norm(rest)/np.linalg.norm(q_current)
    rows = []
    last_following = None
    last_fraction = None
    for fraction in FRACTIONS:
        scale = fraction*unit_scale
        previous = rest_layers[0] + nonlinear.phase_rotate(
            scale*q_previous, -theta)
        current = rest + scale*q_current
        if last_following is None:
            guess = nonlinear.tangent_guess(
                previous, current, rest, rest_layers, factor, 0.0)
        else:
            ratio = fraction/last_fraction
            guess = rest_layers[2] + ratio*(last_following-rest_layers[2])
        following, history, passed = nonlinear.solve(
            previous, current, guess, factor, 0.0)
        row = {
            "amplitude_fraction": fraction,
            "converged": passed,
            "iterations": len(history)-1,
            "final_DEL_RMS": history[-1],
        }
        if passed:
            perturbation = nonlinear.phase_rotate(
                following-rest_layers[2], -theta)
            target = scale*q_target
            retained = qbasis@(qbasis.T@perturbation)
            row.update({
                "native_vs_dirac_relative_error": float(
                    np.linalg.norm(perturbation-target)
                    / np.linalg.norm(target)),
                "outside_packet_lift_fraction": float(
                    np.linalg.norm(perturbation-retained)
                    / max(np.linalg.norm(perturbation), 1e-30)),
            })
            last_following = following
            last_fraction = fraction
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not passed:
            break
    final = rows[-1]
    result = {
        "status": "V4_MIXED_PRONG_DIRAC_AMPLITUDE_CONTINUATION",
        "case_index": args.case_index,
        "flavor": flavor,
        "direction": "diagonal" if axis == 3 else "x",
        "sign": sign,
        "momentum": momentum.tolist(),
        "rows": rows,
        "verdict": (
            "MIXED_PRONG_CONTINUATION_PASS"
            if final["amplitude_fraction"] == 1e-4 and final["converged"]
            and final["native_vs_dirac_relative_error"] < 0.03
            and final["outside_packet_lift_fraction"] < 0.02
            else "MIXED_PRONG_CONTINUATION_OPEN"),
        "claim_boundary": (
            "One native nonlinear update continued from the exact rest "
            "branch. Repeated motion and current/endurance remain."),
    }
    path = HERE/f"mixed_prong_dirac_continuation_case{args.case_index}.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
