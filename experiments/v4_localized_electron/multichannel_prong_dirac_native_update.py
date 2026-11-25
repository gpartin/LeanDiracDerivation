"""Solve the latest prong action's native tangent update and compare to Dirac."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as candidate
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
DT = native_time.DT
SOURCE = "both"


def phase_rotate(raw: np.ndarray, phase: float) -> np.ndarray:
    value = torch.tensor(raw, dtype=torch.float64)
    return precursor.translate(value, 0, 0, phase, 0.0).detach().numpy()


def del_function(previous, current, following):
    return candidate.del_three(previous, current, following, DT, SOURCE)


def jvp(rest_layers, directions) -> np.ndarray:
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in rest_layers)
    tangents = tuple(torch.tensor(value, dtype=torch.float64)
                     for value in directions)
    _value, image = torch.func.jvp(del_function, layers, tangents)
    return image.detach().numpy()


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_layers_np = (
        phase_rotate(rest, -theta), rest, phase_rotate(rest, theta))
    rest_layers = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in rest_layers_np)

    # One action-derived following-layer Jacobian, shared by every packet.
    def following_map(value):
        return del_function(rest_layers[0], rest_layers[1], value)

    jacobian = torch.func.jacrev(following_map)(rest_layers[2])
    following_jacobian = jacobian.detach().numpy()
    factor = lu_factor(following_jacobian)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites,
            dirac.OFFSETS, scales[flavor])
        qbasis = envelope.orth(raw)
        for axis, sign in itertools.product(range(4), (-1, 1)):
            previous, current, following, momentum = dirac.packet(
                axis, sign, alpha, beta, DT)
            delta_previous = phase_rotate(
                raw@dirac.real_coordinates(previous), -theta)
            delta_current = raw@dirac.real_coordinates(current)
            target_lab = phase_rotate(
                raw@dirac.real_coordinates(following), theta)
            rhs = jvp(rest_layers_np, (
                delta_previous, delta_current, np.zeros_like(rest)))
            native_lab = lu_solve(factor, -rhs)
            native_corotating = phase_rotate(native_lab, -theta)
            target_corotating = raw@dirac.real_coordinates(following)
            recovered = qbasis@(qbasis.T@native_corotating)
            target_norm = np.linalg.norm(target_corotating)
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "native_next_vs_dirac_relative_error": float(
                    np.linalg.norm(native_corotating-target_corotating)
                    / target_norm),
                "native_outside_frozen_lift_fraction": float(
                    np.linalg.norm(native_corotating-recovered)
                    / max(np.linalg.norm(native_corotating), 1e-30)),
                "target_DEL_relative_residual": float(
                    np.linalg.norm(rhs+following_jacobian@target_lab)
                    / max(np.linalg.norm(rhs), 1e-30)),
            })

    worst_error = max(row["native_next_vs_dirac_relative_error"]
                      for row in rows)
    worst_leakage = max(row["native_outside_frozen_lift_fraction"]
                        for row in rows)
    result = {
        "status": "V4_MULTICHANNEL_PRONG_DIRAC_NATIVE_UPDATE",
        "source": SOURCE,
        "following_jacobian_condition_number": float(
            np.linalg.cond(following_jacobian)),
        "rows": rows,
        "maximum_native_next_vs_dirac_relative_error": worst_error,
        "maximum_native_outside_frozen_lift_fraction": worst_leakage,
        "verdict": (
            "MULTICHANNEL_PRONG_NATIVE_DIRAC_UPDATE_PASS"
            if worst_error < 1e-3 and worst_leakage < 1e-3 else
            "MULTICHANNEL_PRONG_NATIVE_DIRAC_UPDATE_OPEN"),
        "claim_boundary": (
            "Action-derived one-step linearized update for both copies, both "
            "signs, all axes and a diagonal. Nonlinear repeated motion, "
            "Noether-current matching, endurance and refinement remain."),
    }
    path = HERE/"multichannel_prong_dirac_native_update.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
