"""Test the newest componentwise material action against discrete C19 Dirac.

The Dirac packet supplies only three independently prepared time layers and a
frozen readout.  It is never evaluated inside the native action.  The test
asks whether those layers satisfy the linearized native DEL equation about
the exact localized rest orbit, and decomposes any mismatch into the retained
four-component lift and its complement.
"""

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
import exact_translation_multichannel_prong_gate as candidate
import exact_translation_precursor_linear_gate as precursor


HERE = Path(__file__).resolve().parent
DT = native_time.DT


def phase_rotate(raw: np.ndarray, phase: float) -> np.ndarray:
    tensor = torch.tensor(raw, dtype=torch.float64)
    rotated = precursor.translate(tensor, 0, 0, phase, 0.0)
    return rotated.detach().numpy()


def tangent_image(rest_layers, directions, source: str) -> np.ndarray:
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in rest_layers)
    tangents = tuple(torch.tensor(value, dtype=torch.float64)
                     for value in directions)

    def function(previous, current, following):
        return candidate.del_three(
            previous, current, following, DT, source)

    _value, image = torch.func.jvp(function, layers, tangents)
    return image.detach().numpy()


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_layers = (
        phase_rotate(rest, -theta),
        rest,
        phase_rotate(rest, theta),
    )

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
    for source in ("psi", "both", "chi"):
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
                    phase_rotate(
                        raw@dirac.real_coordinates(previous), -theta),
                    raw@dirac.real_coordinates(current),
                    phase_rotate(
                        raw@dirac.real_coordinates(following), theta),
                )
                images = []
                for selected in range(3):
                    directions = [np.zeros_like(rest) for _ in range(3)]
                    directions[selected] = perturbations[selected]
                    images.append(tangent_image(
                        rest_layers, directions, source))
                residual = sum(images)
                scale = sum(np.linalg.norm(image) for image in images)
                retained = qbasis@(qbasis.T@residual)
                rows.append({
                    "source": source,
                    "flavor": flavor,
                    "direction": "diagonal" if axis == 3 else "xyz"[axis],
                    "sign": sign,
                    "momentum": momentum.tolist(),
                    "relative_DEL_residual": float(
                        np.linalg.norm(residual)/max(scale, 1e-30)),
                    "retained_residual_fraction": float(
                        np.linalg.norm(retained)
                        / max(np.linalg.norm(residual), 1e-30)),
                    "complement_residual_fraction": float(
                        np.linalg.norm(residual-retained)
                        / max(np.linalg.norm(residual), 1e-30)),
                    "layer_image_norms": [
                        float(np.linalg.norm(image)) for image in images],
                })

    source_summaries = {}
    for source in ("psi", "both", "chi"):
        selected = [row for row in rows if row["source"] == source]
        source_summaries[source] = {
            "maximum_relative_DEL_residual": max(
                row["relative_DEL_residual"] for row in selected),
            "mean_relative_DEL_residual": float(np.mean([
                row["relative_DEL_residual"] for row in selected])),
            "maximum_complement_residual_fraction": max(
                row["complement_residual_fraction"] for row in selected),
        }
    best = min(source_summaries,
               key=lambda name: source_summaries[name][
                   "maximum_relative_DEL_residual"])
    worst = source_summaries[best]["maximum_relative_DEL_residual"]
    result = {
        "status": "V4_MULTICHANNEL_PRONG_DIRAC_PACKET_GATE",
        "action": (
            "positive component-preserving material kinetic action with "
            "local phase removed and canonical position potential"),
        "dirac_role": (
            "independent discrete-C19 three-layer comparator and frozen "
            "four-component readout only"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rest_DEL_norms": {
            source: float(np.linalg.norm(candidate.del_three(
                *(torch.tensor(value, dtype=torch.float64)
                  for value in rest_layers), DT, source).detach().numpy()))
            for source in ("psi", "both", "chi")
        },
        "source_summaries": source_summaries,
        "best_source": best,
        "rows": rows,
        "verdict": (
            "MULTICHANNEL_PRONG_DIRAC_TANGENT_PASS"
            if worst < 1e-3 else
            "MULTICHANNEL_PRONG_DIRAC_TANGENT_OPEN"),
        "claim_boundary": (
            "Linearized same-carrier three-layer equation test. A pass would "
            "still require native next-layer evolution, current, two "
            "translations, nonlinear endurance, and refinement before "
            "V4-006 could close."),
    }
    path = HERE/"multichannel_prong_dirac_packet_gate.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
