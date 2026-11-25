"""Discrete-C19 Dirac tangent gate for the per-prong material action."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import exact_translation_per_prong_gate as candidate
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
DT = native_time.DT


def phase_rotate(raw: np.ndarray, phase: float) -> np.ndarray:
    value = torch.tensor(raw, dtype=torch.float64)
    return precursor.translate(value, 0, 0, phase, 0.0).detach().numpy()


def image(rest_layers, directions) -> np.ndarray:
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in rest_layers)
    tangents = tuple(torch.tensor(value, dtype=torch.float64)
                     for value in directions)
    _value, result = torch.func.jvp(
        lambda a, b, c: candidate.del_three(a, b, c, DT),
        layers, tangents)
    return result.detach().numpy()


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_layers = (phase_rotate(rest, -theta), rest,
                   phase_rotate(rest, theta))
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
            perturbations = (
                phase_rotate(raw@dirac.real_coordinates(previous), -theta),
                raw@dirac.real_coordinates(current),
                phase_rotate(raw@dirac.real_coordinates(following), theta),
            )
            images = []
            for selected in range(3):
                directions = [np.zeros_like(rest) for _ in range(3)]
                directions[selected] = perturbations[selected]
                images.append(image(rest_layers, directions))
            residual = sum(images)
            retained = qbasis@(qbasis.T@residual)
            scale = sum(np.linalg.norm(value) for value in images)
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "relative_DEL_residual": float(
                    np.linalg.norm(residual)/max(scale, 1e-30)),
                "complement_residual_fraction": float(
                    np.linalg.norm(residual-retained)
                    / max(np.linalg.norm(residual), 1e-30)),
            })
    worst = max(row["relative_DEL_residual"] for row in rows)
    rest_tensors = tuple(torch.tensor(value, dtype=torch.float64)
                         for value in rest_layers)
    result = {
        "status": "V4_PER_PRONG_DIRAC_PACKET_GATE",
        "rest_DEL_norm": float(torch.linalg.vector_norm(
            candidate.del_three(*rest_tensors, DT)).detach()),
        "maximum_relative_DEL_residual": worst,
        "mean_relative_DEL_residual": float(np.mean([
            row["relative_DEL_residual"] for row in rows])),
        "maximum_complement_residual_fraction": max(
            row["complement_residual_fraction"] for row in rows),
        "rows": rows,
        "verdict": ("PER_PRONG_DIRAC_TANGENT_PASS" if worst < 1e-3
                    else "PER_PRONG_DIRAC_TANGENT_OPEN"),
        "claim_boundary": (
            "Linear tangent gate only; no coefficient was fitted to Dirac."),
    }
    path = HERE/"per_prong_dirac_packet_gate.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
