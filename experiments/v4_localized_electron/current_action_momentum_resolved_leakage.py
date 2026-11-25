"""Resolve the remaining all-center carrier leakage by envelope momentum.

The side-five box supplies 125 allowed Fourier momenta.  At each momentum we
form the same translation-equivariant four-component joint Psi/chi lift and
measure how much of its native acceleration leaves the complete all-center
carrier envelope.  This distinguishes a high-momentum lattice artifact from
a defect already present at the uniform, low-momentum endpoint.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import current_action_localized_envelope_closure as closure
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def momentum_columns(raw: np.ndarray, offsets: list[tuple[int, int, int]],
                     momentum: np.ndarray, components: int) -> np.ndarray:
    """Combine translated real/imag column pairs into one Bloch fiber."""
    per_center = 2 * components
    columns = []
    for component in range(components):
        real = np.zeros(raw.shape[0])
        imag = np.zeros(raw.shape[0])
        for center, offset in enumerate(offsets):
            phase = float(np.dot(momentum, np.asarray(offset)))
            cosine, sine = np.cos(phase), np.sin(phase)
            first = raw[:, center*per_center + 2*component]
            second = raw[:, center*per_center + 2*component + 1]
            real += cosine*first + sine*second
            imag += -sine*first + cosine*second
        columns.extend((real, imag))
    return closure.orth(np.column_stack(columns))


def leakage_channels(leakage: np.ndarray) -> dict:
    shaped = leakage.reshape(5, 5, 5, 4, 3, -1)
    power = np.sum(shaped**2, axis=(0, 1, 2, 5))
    total = max(float(np.sum(power)), 1e-30)
    return {
        "Psi_fraction": float(np.sum(power[:2]) / total),
        "chi_fraction": float(np.sum(power[2:]) / total),
        "component_fractions": [float(value/total)
                                for value in np.sum(power, axis=0)],
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE / "personal_touch_activated_bond_basis_v2_cache.npz")
    matrix = -(bare["stiffness"]
               + additions["hessian_carrier_carrier_chi"]
               + additions["hessian_carrier_carrier_Psi_tight_frame"]
               ) / np.diag(bare["mass"])[:, None]
    psi_frame, chi_frame, sites = closure.carrier_frames(bare["rest"])
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    selected = {
        "plus": 0.00104831247009457,
        "minus": 0.0010251705436500206,
    }
    integer_momenta = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    rows = []
    for flavor, basis in closure.flavor_bases().items():
        raw = closure.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets, selected[flavor])
        all_center = closure.orth(raw)
        flavor_rows = []
        for integer in integer_momenta:
            momentum = 2*np.pi*np.asarray(integer, dtype=float)/5
            q = momentum_columns(raw, offsets, momentum, basis.shape[1])
            response = matrix @ q
            leakage = response - all_center @ (all_center.T @ response)
            flavor_rows.append({
                "integer_momentum": list(integer),
                "momentum_norm": float(np.linalg.norm(momentum)),
                "relative_outside_all_center_envelope": float(
                    np.linalg.norm(leakage)/max(np.linalg.norm(response), 1e-30)),
                "leakage_channels": leakage_channels(leakage),
            })
        zero = next(row for row in flavor_rows
                    if row["integer_momentum"] == [0, 0, 0])
        first_shell = [row for row in flavor_rows
                       if sum(value*value for value in row["integer_momentum"]) == 1]
        rows.append({
            "flavor": flavor,
            "zero_momentum": zero,
            "smallest_nonzero_momentum_max_residual": float(max(
                row["relative_outside_all_center_envelope"]
                for row in first_shell)),
            "smallest_nonzero_momentum_min_residual": float(min(
                row["relative_outside_all_center_envelope"]
                for row in first_shell)),
            "all_momentum_max_residual": float(max(
                row["relative_outside_all_center_envelope"]
                for row in flavor_rows)),
            "all_momentum_min_residual": float(min(
                row["relative_outside_all_center_envelope"]
                for row in flavor_rows)),
            "rows": flavor_rows,
        })
    zero_worst = max(row["zero_momentum"][
        "relative_outside_all_center_envelope"] for row in rows)
    result = {
        "status": "V4_CURRENT_ACTION_MOMENTUM_RESOLVED_CARRIER_LEAKAGE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": 5,
        "rows": rows,
        "worst_zero_momentum_residual": float(zero_worst),
        "verdict": (
            "LEAKAGE_VANISHES_AT_ZERO_MOMENTUM"
            if zero_worst < 1e-8 else
            "LEAKAGE_PERSISTS_AT_ZERO_MOMENTUM"),
        "claim_boundary": (
            "The side-five zero mode is the uniform envelope endpoint, not a "
            "box-refined continuum extrapolation. Persistence there rejects "
            "the explanation that the observed defect is purely a high-k "
            "artifact on this branch; larger boxes remain a separate test."),
    }
    (HERE / "current_action_momentum_resolved_leakage.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
