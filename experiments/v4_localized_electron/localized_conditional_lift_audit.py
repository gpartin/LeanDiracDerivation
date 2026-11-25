"""Audit the unproved lift from cubical fibers to physical corner sites.

Lean's cubical Clifford coordinates form an internal eight-coordinate fiber.
The movement experiments tentatively applied that operator to the eight
physical carrier sites.  This diagnostic measures whether that identification
is compatible with a literal translated carrier and whether summing both
flavor copies repairs any mismatch.  It changes no action and proves no
physical embedding.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import conditional_edge_native_gate as conditional
import cubical_edge_shift_channels as cube
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5


def core_sites() -> list[tuple[int, int, int]]:
    center = SIDE // 2 - 1
    return [tuple(center + bit for bit in corner)
            for corner in itertools.product((0, 1), repeat=3)]


def project(field: np.ndarray, projector: np.ndarray) -> np.ndarray:
    return conditional.apply_fiber_operator(field, projector)


def support_relative(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first-second)
                 / max(np.linalg.norm(second), 1e-30))


def classify(field: np.ndarray, vacuum: np.ndarray,
             axis: int, sign: int) -> dict:
    identity = np.eye(8, dtype=complex)
    flavor = cube.flavor()
    plus = (identity+flavor)/2
    minus = (identity-flavor)/2
    deviation = field-vacuum

    desired = vacuum + np.roll(deviation, sign, axis=axis)
    conditional_both = vacuum + conditional.conditional_step(
        deviation, axis, sign)
    plus_deviation = project(deviation, plus)
    minus_deviation = project(deviation, minus)
    conditional_plus = conditional.conditional_step(
        plus_deviation, axis, sign)
    conditional_minus = conditional.conditional_step(
        minus_deviation, axis, sign)
    copy_sum = vacuum + conditional_plus + conditional_minus

    source = set(core_sites())
    destination = {
        tuple((coordinate + (sign if index == axis else 0)) % SIDE
              for index, coordinate in enumerate(site))
        for site in source
    }
    shared = sorted(source & destination)
    shared_conditional = np.stack([conditional_both[site] for site in shared])
    shared_desired = np.stack([desired[site] for site in shared])
    return {
        "axis": axis,
        "sign": sign,
        "shared_physical_corner_count": len(shared),
        "conditional_vs_rigid_full_field_relative_defect": support_relative(
            conditional_both, desired),
        "conditional_vs_rigid_shared_face_relative_defect": support_relative(
            shared_conditional, shared_desired),
        "two_copy_sum_vs_direct_conditional_defect": support_relative(
            copy_sum, conditional_both),
        "plus_copy_norm_fraction": float(
            np.linalg.norm(conditional_plus)**2
            / max(np.linalg.norm(conditional_plus)**2
                  + np.linalg.norm(conditional_minus)**2, 1e-30)),
        "minus_copy_norm_fraction": float(
            np.linalg.norm(conditional_minus)**2
            / max(np.linalg.norm(conditional_plus)**2
                  + np.linalg.norm(conditional_minus)**2, 1e-30)),
    }


def main() -> None:
    rank.lock_inertia()
    amplitude_sq = rank.R**2/(2+54/rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    zero = np.zeros_like(psi)
    chi_vacuum = np.zeros_like(chi)
    chi_vacuum[..., 2] = rank.R
    result = {
        "status": "V4_LOCALIZED_CONDITIONAL_LIFT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "lean_source_boundary": (
            "LeanDirac/V2/NativeCubicalCompression.lean states that the eight "
            "coordinates are an internal cubical fiber and are not asserted "
            "to be eight separate physical lattice sites."),
        "psi_rows": [classify(psi, zero, axis, sign)
                     for axis in range(3) for sign in (-1, 1)],
        "chi_rows": [classify(chi, chi_vacuum, axis, sign)
                     for axis in range(3) for sign in (-1, 1)],
        "verdict": "PHYSICAL_SITE_LIFT_NOT_ESTABLISHED",
        "root_cause": (
            "The derived conditional shift is exact in the internal cubical "
            "fiber, but the present localized tests identify that fiber with "
            "overlapping physical corners without an intertwining embedding. "
            "Adding both flavor copies only reconstructs the same internal "
            "conditional shift; it cannot repair a lift mismatch."),
        "corrective_action": (
            "Derive one localized embedding E from full Psi/chi phase space "
            "to the cubical fiber and prove a translation intertwiner on "
            "overlapping cubes. Only then define the native edge-handoff "
            "target and resume action repair against its physical residual."),
        "claim_boundary": (
            "This detects a missing proof/measurement bridge. It does not "
            "disprove edge hopping or prove that the two copies are physical "
            "particles."),
    }
    (HERE / "localized_conditional_lift_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
