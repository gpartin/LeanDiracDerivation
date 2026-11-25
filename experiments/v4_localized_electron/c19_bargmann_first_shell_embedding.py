"""Test whether Bargmann-connection leakage is a translated carrier tail.

The frozen eight-component audit compared the exact connection Hessian with
one directional carrier ramp.  A moving localized carrier is instead a
four-component internal fiber over several lattice centers.  This audit keeps
the action fixed and enlarges only the diagnostic embedding, first to the C19
neighbor centers and then to every center of the periodic side-five box.

No projection is used in evolution.  Orthogonal projection is only the
readout used to measure whether the native Hessian image belongs to the
translated carrier bundle.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import qr

import c19_bargmann_connection_copy_gate as gate
import c19_triangle_same_embedding_closure as embedding
import localized_carrier_contrast_inverse_launch as inverse
import localized_stability as native


HERE = Path(__file__).resolve().parent


def orth(raw: np.ndarray) -> np.ndarray:
    q, r, _ = qr(raw, mode="economic", pivoting=True, check_finite=False)
    diagonal = abs(np.diag(r))
    tolerance = 1.0e-11 * max(float(diagonal[0]), 1.0)
    return q[:, diagonal > tolerance]


def translated_bundle(flavor: str, psi_phase: complex,
                      offsets: list[tuple[int, int, int]]) -> np.ndarray:
    psi_profile, chi_profile = inverse.contrast.carrier_contrast()
    basis = inverse.local.flavor_bases()[flavor]
    raw = np.column_stack([
        column
        for offset in offsets
        for column in inverse.profile_columns(
            basis, psi_phase * psi_profile, chi_profile, offset)
    ])
    return orth(raw)


def project_residual(images: np.ndarray, basis: np.ndarray) -> float:
    residual = images - basis @ (basis.T @ images)
    return float(np.linalg.norm(residual)
                 / max(np.linalg.norm(images), 1.0e-300))


def support_power(raw: np.ndarray) -> dict[str, float]:
    """Separate core, C19-neighbor-center shell, and remaining response."""
    psi, chi = inverse.rank.native.unpack(raw, inverse.SIDE)
    power = inverse.rank.native.norm2(psi) + inverse.rank.B * (
        inverse.rank.native.norm2(chi))
    core = np.zeros((inverse.SIDE,) * 3, dtype=bool)
    shell = np.zeros_like(core)
    for bits in itertools.product((0, 1), repeat=3):
        core[tuple((inverse.ORIGIN + bit) % inverse.SIDE for bit in bits)] = True
    for offset, _weight in native.DIRECTIONS:
        for bits in itertools.product((0, 1), repeat=3):
            site = tuple((inverse.ORIGIN + offset[axis] + bits[axis])
                         % inverse.SIDE for axis in range(3))
            shell[site] = True
    shell &= ~core
    total = max(float(np.sum(power)), 1.0e-300)
    return {
        "core": float(np.sum(power[core]) / total),
        "c19_first_shell": float(np.sum(power[shell]) / total),
        "outside_c19_first_shell": float(np.sum(power[~(core | shell)]) / total),
    }


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))

    c19_offsets = [(0, 0, 0)] + [offset for offset, _ in native.DIRECTIONS]
    all_offsets = list(itertools.product(range(-2, 3), repeat=3))
    phases = {
        "previous": np.exp(-1j * theta),
        "following": np.exp(1j * theta),
    }
    bundles = {}
    for extent, offsets in (("c19", c19_offsets), ("all_centers", all_offsets)):
        for layer, phase in phases.items():
            for flavor in ("plus", "minus"):
                bundles[(extent, layer, flavor)] = translated_bundle(
                    flavor, phase, offsets)
            bundles[(extent, layer, "both")] = orth(np.column_stack([
                bundles[(extent, layer, "plus")],
                bundles[(extent, layer, "minus")],
            ]))

    rows = []
    for axis in range(3):
        labels, targets = embedding.component_targets(
            axis, np.exp(1j * theta))
        images = gate.images(*layers, targets)
        split = {
            "plus": images[:, :4],
            "minus": images[:, 4:],
            "both": images,
        }
        row = {
            "axis": axis,
            "image_norm": float(np.linalg.norm(images)),
            "support_power": {
                flavor: support_power(np.sqrt(np.sum(values**2, axis=1)))
                for flavor, values in split.items()
            },
            "projection_residuals": {},
        }
        for extent in ("c19", "all_centers"):
            for flavor in ("plus", "minus"):
                key = f"{extent}_{flavor}_own"
                row["projection_residuals"][key] = project_residual(
                    split[flavor], bundles[(extent, "previous", flavor)])
                opposite = "minus" if flavor == "plus" else "plus"
                row["projection_residuals"][
                    f"{extent}_{flavor}_opposite"] = project_residual(
                        split[flavor],
                        bundles[(extent, "previous", opposite)])
            row["projection_residuals"][f"{extent}_both"] = project_residual(
                images, bundles[(extent, "previous", "both")])
        # A following-layer target must itself remain in the same translated
        # bundle; this guards against a misleading previous-layer-only fit.
        row["following_target_residuals"] = {
            flavor: project_residual(
                targets[:, :4] if flavor == "plus" else targets[:, 4:],
                bundles[("c19", "following", flavor)])
            for flavor in ("plus", "minus")
        }
        rows.append(row)

    worst_c19_both = max(
        row["projection_residuals"]["c19_both"] for row in rows)
    worst_all_both = max(
        row["projection_residuals"]["all_centers_both"] for row in rows)
    worst_own = max(
        max(row["projection_residuals"]["c19_plus_own"],
            row["projection_residuals"]["c19_minus_own"])
        for row in rows)
    maximum_shell_power = max(
        row["support_power"][flavor]["c19_first_shell"]
        for row in rows for flavor in ("plus", "minus"))
    if maximum_shell_power < 1.0e-10:
        verdict = "BARGMANN_LEAKAGE_IS_INTERNAL_CORE_NOT_FIRST_SHELL"
    elif worst_own < 1.0e-6:
        verdict = "BARGMANN_RESPONSE_IS_SINGLE_COPY_TRANSLATED_CARRIER_TAIL"
    elif worst_c19_both < 1.0e-6:
        verdict = "BARGMANN_RESPONSE_REQUIRES_BOTH_COPY_TRANSLATED_BUNDLE"
    else:
        verdict = "FIRST_SHELL_DOES_NOT_CLOSE_BARGMANN_RESPONSE"
    result = {
        "status": "V4_C19_BARGMANN_FIRST_SHELL_EMBEDDING",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": inverse.SIDE,
        "c19_center_count": len(c19_offsets),
        "all_center_count": len(all_offsets),
        "bundle_ranks": {
            f"{extent}_{layer}_{flavor}": int(
                bundles[(extent, layer, flavor)].shape[1])
            for extent in ("c19", "all_centers")
            for layer in ("previous", "following")
            for flavor in ("plus", "minus", "both")
        },
        "rows": rows,
        "worst_c19_both_copy_residual": worst_c19_both,
        "worst_c19_single_copy_residual": worst_own,
        "worst_all_center_both_copy_residual": worst_all_both,
        "maximum_c19_first_shell_power": maximum_shell_power,
        "verdict": verdict,
        "claim_boundary": (
            "One-Hessian-image diagnostic for the rest-preserving Bargmann "
            "connection. A small residual supports a translated carrier "
            "bundle but does not prove repeated invariance, nonlinear "
            "stability, coefficient provenance, or V4-006."),
    }
    output = HERE / "c19_bargmann_first_shell_embedding.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "bundle_ranks": result["bundle_ranks"],
        "worst_c19_both_copy_residual": worst_c19_both,
        "worst_c19_single_copy_residual": worst_own,
        "worst_all_center_both_copy_residual": worst_all_both,
        "maximum_c19_first_shell_power": maximum_shell_power,
        "support": [row["support_power"] for row in rows],
        "output": str(output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
