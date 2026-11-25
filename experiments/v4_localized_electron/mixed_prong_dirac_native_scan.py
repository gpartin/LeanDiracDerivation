"""Scan geometric mixtures of common and relative per-prong material motion.

Both endpoint actions are positive component-preserving kinetic actions and
contain the same position potential.  Convex mixing therefore preserves one
copy of the ordinary kinetic and potential terms.  The scan is diagnostic:
Dirac does not set a coefficient.
"""

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
import exact_translation_multichannel_prong_gate as common
import exact_translation_per_prong_gate as relative
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
DT = native_time.DT
ETAS = (0.0, 0.25, 1/3, 0.5, 0.75, 1.0)


def phase_rotate(raw: np.ndarray, phase: float) -> np.ndarray:
    value = torch.tensor(raw, dtype=torch.float64)
    return precursor.translate(value, 0, 0, phase, 0.0).detach().numpy()


def endpoint_del(kind, previous, current, following):
    if kind == "common":
        return common.del_three(
            previous, current, following, DT, "both")
    return relative.del_three(previous, current, following, DT)


def endpoint_jvp(kind, rest_layers, directions) -> np.ndarray:
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in rest_layers)
    tangents = tuple(torch.tensor(value, dtype=torch.float64)
                     for value in directions)
    _value, result = torch.func.jvp(
        lambda a, b, c: endpoint_del(kind, a, b, c), layers, tangents)
    return result.detach().numpy()


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_layers_np = (
        phase_rotate(rest, -theta), rest, phase_rotate(rest, theta))
    rest_layers = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in rest_layers_np)

    jacobians = {}
    for kind in ("common", "relative"):
        jacobians[kind] = torch.func.jacrev(
            lambda value, selected=kind: endpoint_del(
                selected, rest_layers[0], rest_layers[1], value))(
                    rest_layers[2]).detach().numpy()

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

    cases = []
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites,
            dirac.OFFSETS, scales[flavor])
        qbasis = envelope.orth(raw)
        for axis, sign in itertools.product(range(4), (-1, 1)):
            previous, current, following, momentum = dirac.packet(
                axis, sign, alpha, beta, DT)
            directions = (
                phase_rotate(raw@dirac.real_coordinates(previous), -theta),
                raw@dirac.real_coordinates(current),
                np.zeros_like(rest),
            )
            rhs = {
                kind: endpoint_jvp(kind, rest_layers_np, directions)
                for kind in ("common", "relative")
            }
            cases.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "target": raw@dirac.real_coordinates(following),
                "qbasis": qbasis,
                "rhs": rhs,
            })

    scan = []
    for eta in ETAS:
        jacobian = ((1-eta)*jacobians["common"]
                    + eta*jacobians["relative"])
        factor = lu_factor(jacobian)
        rows = []
        for case in cases:
            rhs = ((1-eta)*case["rhs"]["common"]
                   + eta*case["rhs"]["relative"])
            native_lab = lu_solve(factor, -rhs)
            native = phase_rotate(native_lab, -theta)
            retained = case["qbasis"]@(case["qbasis"].T@native)
            target = case["target"]
            rows.append({
                "flavor": case["flavor"],
                "direction": case["direction"],
                "sign": case["sign"],
                "native_next_vs_dirac_relative_error": float(
                    np.linalg.norm(native-target)/np.linalg.norm(target)),
                "native_outside_frozen_lift_fraction": float(
                    np.linalg.norm(native-retained)
                    / max(np.linalg.norm(native), 1e-30)),
            })
        scan.append({
            "relative_prong_weight": eta,
            "common_channel_weight": 1-eta,
            "geometric_interpretation": (
                "one common plus three relative channels equally counted"
                if eta == 0.75 else
                "equal aggregate common and relative weights"
                if eta == 0.5 else "diagnostic control"),
            "maximum_native_next_vs_dirac_relative_error": max(
                row["native_next_vs_dirac_relative_error"] for row in rows),
            "mean_native_next_vs_dirac_relative_error": float(np.mean([
                row["native_next_vs_dirac_relative_error"] for row in rows])),
            "maximum_native_outside_frozen_lift_fraction": max(
                row["native_outside_frozen_lift_fraction"] for row in rows),
            "rows": rows,
        })
    best = min(scan, key=lambda row:
               row["maximum_native_next_vs_dirac_relative_error"])
    result = {
        "status": "V4_MIXED_PRONG_DIRAC_NATIVE_SCAN",
        "coefficient_policy": (
            "geometric rational controls only; no coefficient admitted from "
            "the Dirac comparison"),
        "scan": scan,
        "best_diagnostic_weight": best["relative_prong_weight"],
        "best_diagnostic_maximum_error": best[
            "maximum_native_next_vs_dirac_relative_error"],
        "verdict": (
            "GEOMETRIC_MIXED_PRONG_NATIVE_UPDATE_PASS"
            if any(row["geometric_interpretation"] != "diagnostic control"
                   and row["maximum_native_next_vs_dirac_relative_error"] < 0.03
                   and row["maximum_native_outside_frozen_lift_fraction"] < 0.02
                   for row in scan)
            else "GEOMETRIC_MIXED_PRONG_NATIVE_UPDATE_OPEN"),
        "claim_boundary": (
            "One-step tangent scan. No weight is admitted until derived "
            "independently and passed through nonlinear/current/endurance "
            "gates."),
    }
    path = HERE/"mixed_prong_dirac_native_scan.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "scan": [{key: value for key, value in row.items()
                  if key != "rows"} for row in scan],
        "best_diagnostic_weight": result["best_diagnostic_weight"],
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
