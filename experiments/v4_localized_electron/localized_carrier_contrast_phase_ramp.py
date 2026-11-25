"""Screen native neighboring-center phase ramps for edge transport.

Momentum is not a local phase rotation; it is an odd phase gradient across
neighboring carrier centers.  This audit builds that gradient from the exact
carrier-minus-vacuum contrast, separately for the two cubical copies and for
their joint/relative combinations.  The unchanged native DEL equation selects
the previous layer, after which the Psi Noether-center response is measured.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_fixed_charge_launch as fixed
import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def copy_amplitudes() -> dict[str, np.ndarray]:
    flavors = inverse.local.flavor_bases()
    ones = np.ones(8, dtype=complex)
    return {name: basis@np.linalg.lstsq(basis, ones, rcond=None)[0]
            for name, basis in flavors.items()}


def placed_copy(amplitudes: np.ndarray, offset: tuple[int, int, int],
                psi_phase: complex) -> np.ndarray:
    psi_profile, chi_profile = inverse.contrast.carrier_contrast()
    dpsi = np.zeros((inverse.SIDE,)*3+(3,), dtype=complex)
    dchi = np.zeros_like(dpsi)
    for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
        site = tuple(inverse.ORIGIN+offset[axis]+bits[axis]
                     for axis in range(3))
        dpsi[site] += amplitudes[corner]*psi_phase*psi_profile[corner]
        dchi[site] += amplitudes[corner]*chi_profile[corner]
    return inverse.rank.pack(dpsi, dchi)


def phase_ramps(axis: int, psi_phase: complex) -> dict[str, np.ndarray]:
    amplitudes = copy_amplitudes()
    offset_plus = tuple(1 if index == axis else 0 for index in range(3))
    offset_minus = tuple(-1 if index == axis else 0 for index in range(3))
    copy_rows = {}
    for name, values in amplitudes.items():
        odd_difference = (
            placed_copy(values, offset_plus, psi_phase)
            - placed_copy(values, offset_minus, psi_phase))/2
        psi, chi = inverse.rank.native.unpack(
            odd_difference, inverse.SIDE)
        copy_rows[name] = inverse.rank.pack(1j*psi, 1j*chi)
    return {
        "plus_copy_ramp": copy_rows["plus"],
        "minus_copy_ramp": copy_rows["minus"],
        "joint_copy_ramp": copy_rows["plus"]+copy_rows["minus"],
        "relative_copy_ramp": copy_rows["plus"]-copy_rows["minus"],
    }


def center_displacement(previous: np.ndarray, current: np.ndarray,
                        following: np.ndarray, launch: np.ndarray,
                        target: np.ndarray) -> np.ndarray:
    current_center = fixed.center_derivative(
        previous, current, launch, None, "Psi")
    following_center = fixed.center_derivative(
        current, following, None, target, "Psi")
    return following_center-current_center


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(inverse.torch.tensor(value, dtype=inverse.torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    previous_block = inverse.batched_jvp(
        "previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        for name, target in phase_ramps(
                axis, np.exp(1j*theta)).items():
            target_image = inverse.batched_jvp(
                "following", base, target[:, None])[:, 0]
            launch = lu_solve(factor, -target_image)
            residual = previous_block@launch+target_image
            charges = fixed.charge_derivatives(
                previous, current, following, launch, target)
            displacement = center_displacement(
                previous, current, following, launch, target)
            rows.append({
                "axis": axis,
                "phase_ramp_family": name,
                "DEL_relative_residual": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(target_image), 1e-30)),
                "native_charge_derivatives": charges.tolist(),
                "Psi_center_displacement_response": [float(value)
                                                     for value in displacement],
                "requested_axis_response": float(displacement[axis]),
                "transverse_response_norm": float(np.linalg.norm(
                    np.delete(displacement, axis))),
                "longitudinal_to_transverse_ratio": float(
                    abs(displacement[axis])
                    / max(np.linalg.norm(np.delete(displacement, axis)),
                          1e-30)),
                "launch_support_power_fractions": full.support_fractions(
                    launch),
            })
    best = max(rows, key=lambda row: row[
        "longitudinal_to_transverse_ratio"])
    result = {
        "status": "V4_CARRIER_CONTRAST_PHASE_RAMP_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "construction": (
            "odd native phase gradient across the two neighboring carrier "
            "centers, built separately from each cubical copy"),
        "rows": rows,
        "best_family": best["phase_ramp_family"],
        "best_axis": best["axis"],
        "best_longitudinal_to_transverse_ratio": best[
            "longitudinal_to_transverse_ratio"],
        "verdict": (
            "NATIVE_COPY_PHASE_RAMP_HAS_LONGITUDINAL_RESPONSE" if
            best["longitudinal_to_transverse_ratio"] > 10 else
            "NATIVE_COPY_PHASE_RAMP_RESPONSE_INSUFFICIENT"),
        "claim_boundary": (
            "This is a linear momentum-response screen. It is not a finite "
            "hop, fixed-charge branch, repeated translation, or Dirac-current "
            "match."),
        "corrective_action": (
            "If a ramp is longitudinal, combine it with the native common "
            "phase tangents to enforce both charge constraints and retest. "
            "If all ramps are transverse or null, the action lacks the "
            "required phase-gradient response."),
    }
    (HERE / "localized_carrier_contrast_phase_ramp.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
