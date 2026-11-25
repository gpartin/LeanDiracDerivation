"""Screen the native common/relative Psi--chi phase-space bond channels.

The density-oriented common connection uses Z=chi tensor Psi.  Its equally
native conjugate companion uses Z=chi tensor conjugate(Psi), and therefore
transmits the relative/counter-rotating phase information that the common
channel cannot distinguish.  This script measures literal Psi site transfer;
it does not admit a coefficient or install a new action.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_geometry_bifurcation_solve as solve
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import discrete_site_transfer_observable_gate as observable


HERE = Path(__file__).resolve().parent
FACTORS = (0.0, 12.0, 19.0, 32.0, 48.0)


def evolve_case(matrices, raw, alpha, beta, axis, sign, dt, steps):
    mass0, gyro0, stiffness0 = matrices
    mass, gyro, stiffness = map(
        observable.sparse, (mass0, gyro0, stiffness0))
    factor = splu((mass / dt + gyro / 2).tocsc())
    current_block = (dt * stiffness - 2 * mass / dt).tocsc()
    previous_block = (mass / dt - gyro / 2).tocsc()
    operators = observable.propagators(alpha, beta, dt)
    previous_spinor, current_spinor, _next, momentum = gate.packet(
        axis, sign, alpha, beta, dt)
    q_previous = raw @ gate.real_coordinates(previous_spinor)
    q_current = raw @ gate.real_coordinates(current_spinor)
    initial_native_center = None
    initial_target_center = None
    maximum_density_error = 0.0
    maximum_center_error = 0.0
    first_five_percent_cycle = None
    final_density_error = 0.0
    finite_evolution = True
    for step in range(steps + 1):
        if (not np.all(np.isfinite(q_current))
                or float(np.max(abs(q_current))) > 1e150):
            finite_evolution = False
            maximum_density_error = 1e300
            maximum_center_error = 1e300
            final_density_error = 1e300
            break
        target = raw @ gate.real_coordinates(current_spinor)
        native_density = observable.psi_pixel_density(q_current)
        target_density = observable.psi_pixel_density(target)
        density_error = observable.normalized_density_error(
            observable.c19_average(native_density),
            observable.c19_average(target_density))
        native_center, _ = observable.current_bridge.center(native_density)
        target_center, _ = observable.current_bridge.center(target_density)
        if initial_native_center is None:
            initial_native_center = native_center
            initial_target_center = target_center
        native_shift = observable.current_bridge.wrapped_shift(
            initial_native_center, native_center)
        target_shift = observable.current_bridge.wrapped_shift(
            initial_target_center, target_center)
        center_error = float(np.linalg.norm(native_shift - target_shift))
        maximum_density_error = max(maximum_density_error, density_error)
        maximum_center_error = max(maximum_center_error, center_error)
        if density_error >= 0.05 and first_five_percent_cycle is None:
            first_five_percent_cycle = (
                step * dt * native_time.CHI0 / (2 * np.pi))
        if step == steps:
            final_density_error = density_error
            break
        q_following = factor.solve(
            -(current_block @ q_current + previous_block @ q_previous))
        if (not np.all(np.isfinite(q_following))
                or float(np.max(abs(q_following))) > 1e150):
            finite_evolution = False
            maximum_density_error = 1e300
            maximum_center_error = 1e300
            final_density_error = 1e300
            break
        q_previous, q_current = q_current, q_following
        previous_spinor, current_spinor = (
            current_spinor,
            observable.advance_dirac(current_spinor, operators))
    return {
        "direction": "diagonal" if axis == 3 else "x",
        "sign": sign,
        "momentum": momentum.tolist(),
        "first_five_percent_cycle": first_five_percent_cycle,
        "maximum_C19_averaged_Psi_pixel_density_relative_error": (
            maximum_density_error),
        "final_C19_averaged_Psi_pixel_density_relative_error": (
            final_density_error),
        "maximum_Psi_pixel_center_shift_error": maximum_center_error,
        "finite_evolution": finite_evolution,
    }


def main() -> None:
    bare, _ = action.expected_blocks()
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    base = ablation.blocks()["base_plus_density_connection"]
    cache = np.load(HERE / "density_oriented_pair_connection_matrices.npz")
    relative = tuple(cache[f"{kind}_relative_pair_connection"]
                     for kind in ("mass", "gyro", "stiffness"))
    natural = native_time.B * native_time.CHI0
    dt = native_time.DT / 16
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))

    prepared = {}
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        prepared[flavor] = (raw, alpha, beta)
    screen_rows = []
    for relative_factor in FACTORS:
        matrices = tuple(base[index] + natural * relative_factor
                         * relative[index] for index in range(3))
        for flavor, (raw, alpha, beta) in prepared.items():
            case = evolve_case(
                matrices, raw, alpha, beta, 0, 1, dt, steps)
            screen_rows.append({
                "flavor": flavor,
                "relative_factor_in_B_chi0_units": relative_factor,
                **case,
            })
    factor_rows = []
    for relative_factor in FACTORS:
        cases = [row for row in screen_rows if row[
            "relative_factor_in_B_chi0_units"] == relative_factor]
        factor_rows.append({
            "relative_factor_in_B_chi0_units": relative_factor,
            "finite_for_both_copies": all(
                row["finite_evolution"] for row in cases),
            "worst_copy_density_error": max(row[
                "maximum_C19_averaged_Psi_pixel_density_relative_error"]
                for row in cases),
            "worst_copy_center_error": max(row[
                "maximum_Psi_pixel_center_shift_error"] for row in cases),
        })
    best = min((row for row in factor_rows
                if row["finite_for_both_copies"]), key=lambda row: row[
                    "worst_copy_density_error"])

    held_rows = []
    best_factor = best["relative_factor_in_B_chi0_units"]
    matrices = tuple(base[index] + natural * best_factor * relative[index]
                     for index in range(3))
    for flavor, (raw, alpha, beta) in prepared.items():
        for axis, sign in ((0, -1), (0, 1), (3, 1)):
            held_rows.append({
                "flavor": flavor,
                **evolve_case(
                    matrices, raw, alpha, beta, axis, sign, dt, steps),
            })

    zero = next(row for row in factor_rows
                if row["relative_factor_in_B_chi0_units"] == 0.0)
    worst_held_density = max(row[
        "maximum_C19_averaged_Psi_pixel_density_relative_error"]
        for row in held_rows)
    worst_held_center = max(row["maximum_Psi_pixel_center_shift_error"]
                            for row in held_rows)
    result = {
        "status": "V4_DISCRETE_SITE_TRANSFER_PAIR_CHANNEL_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "common_channel": "density contrast times chi tensor Psi one-form",
        "relative_channel": (
            "density contrast times chi tensor conjugate(Psi) one-form"),
        "natural_scale": natural,
        "natural_scale_provenance": "B times chi0",
        "screen_rows": screen_rows,
        "factor_minimax_rows": factor_rows,
        "best_minimax_factor": best_factor,
        "baseline_worst_copy_density_error": zero[
            "worst_copy_density_error"],
        "best_minimax_worst_copy_density_error": best[
            "worst_copy_density_error"],
        "held_rows_at_best_screen_factor": held_rows,
        "worst_held_density_error": worst_held_density,
        "worst_held_center_error": worst_held_center,
        "verdict": (
            "RELATIVE_PHASE_CHANNEL_SITE_TRANSFER_PASS"
            if worst_held_density < 0.05 and worst_held_center < 0.01
            else "RELATIVE_PHASE_CHANNEL_DOES_NOT_CLOSE_SITE_TRANSFER"),
        "claim_boundary": (
            "The factor screen is a diagnostic against both copies in one "
            "axial direction. "
            "The best factor is not coefficient provenance and is not "
            "admitted into the action. Held rows cover both copies, axial "
            "signs and a diagonal for one linear cycle on side five."),
    }
    (HERE / "discrete_site_transfer_pair_channel_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("screen_rows", "factor_minimax_rows",
                                     "held_rows_at_best_screen_factor")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
