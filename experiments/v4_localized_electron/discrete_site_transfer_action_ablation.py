"""Ablate research-action terms using the no-intravoxel motion observable.

This is a diagnostic, not an action search.  It evolves the same extended
plus-copy x-positive packet with each already-declared action ablation and
compares literal Psi site density with the independently evolved discrete
C19 Dirac packet.  No fractional carrier position or compact-shape shift is
used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import discrete_site_transfer_observable_gate as observable


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare, _ = action.expected_blocks()
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    plus_case = next(
        item
        for row in prior["rows"] if row["flavor"] == "plus"
        for item in row["cases"] if item["envelope"] == "all_125_centers")
    chi_scale = complex(
        plus_case["chi_scale_real"], plus_case["chi_scale_imag"])

    basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(basis)
    dt = native_time.DT / 16
    operators = observable.propagators(alpha, beta, dt)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, chi_scale)
    dirac_previous0, dirac_current0, _next, momentum = gate.packet(
        0, 1, alpha, beta, dt)
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))
    sample_steps = set(np.linspace(0, steps, 65, dtype=int).tolist())

    rows = []
    for name, (mass0, gyro0, stiffness0) in ablation.blocks().items():
        mass, gyro, stiffness = map(
            observable.sparse, (mass0, gyro0, stiffness0))
        factor = splu((mass / dt + gyro / 2).tocsc())
        current_block = (dt * stiffness - 2 * mass / dt).tocsc()
        previous_block = (mass / dt - gyro / 2).tocsc()
        dirac_previous = dirac_previous0.copy()
        dirac_current = dirac_current0.copy()
        q_previous = raw @ gate.real_coordinates(dirac_previous)
        q_current = raw @ gate.real_coordinates(dirac_current)
        initial_native_center = None
        initial_target_center = None
        samples = []
        for step in range(steps + 1):
            if step in sample_steps:
                target_current = raw @ gate.real_coordinates(dirac_current)
                native_density = observable.psi_pixel_density(q_current)
                target_density = observable.psi_pixel_density(target_current)
                native_center, _ = observable.current_bridge.center(
                    native_density)
                target_center, _ = observable.current_bridge.center(
                    target_density)
                if initial_native_center is None:
                    initial_native_center = native_center
                    initial_target_center = target_center
                native_shift = observable.current_bridge.wrapped_shift(
                    initial_native_center, native_center)
                target_shift = observable.current_bridge.wrapped_shift(
                    initial_target_center, target_center)
                samples.append({
                    "step": step,
                    "cycles": step * dt * native_time.CHI0 / (2 * np.pi),
                    "C19_averaged_Psi_pixel_density_relative_error": (
                        observable.normalized_density_error(
                            observable.c19_average(native_density),
                            observable.c19_average(target_density))),
                    "Psi_pixel_center_shift_error": float(np.linalg.norm(
                        native_shift - target_shift)),
                })
            if step == steps:
                break
            q_following = factor.solve(
                -(current_block @ q_current
                  + previous_block @ q_previous))
            q_previous, q_current = q_current, q_following
            dirac_previous, dirac_current = (
                dirac_current,
                observable.advance_dirac(dirac_current, operators))

        first_five = next((sample for sample in samples
                           if sample[
                               "C19_averaged_Psi_pixel_density_relative_error"]
                           >= 0.05), None)
        rows.append({
            "action_ablation": name,
            "first_five_percent_cycle": (
                first_five["cycles"] if first_five else None),
            "maximum_C19_averaged_Psi_pixel_density_relative_error": max(
                sample[
                    "C19_averaged_Psi_pixel_density_relative_error"]
                for sample in samples),
            "final_C19_averaged_Psi_pixel_density_relative_error": samples[-1][
                "C19_averaged_Psi_pixel_density_relative_error"],
            "maximum_Psi_pixel_center_shift_error": max(
                sample["Psi_pixel_center_shift_error"]
                for sample in samples),
            "samples": samples,
        })

    best = min(rows, key=lambda row:
               row["maximum_C19_averaged_Psi_pixel_density_relative_error"])
    result = {
        "status": "V4_DISCRETE_SITE_TRANSFER_ACTION_ABLATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "flavor": "plus",
        "direction": "x",
        "sign": 1,
        "momentum": momentum.tolist(),
        "dt": dt,
        "cycles": 1.0,
        "rows": rows,
        "best_existing_ablation": best["action_ablation"],
        "best_maximum_density_error": best[
            "maximum_C19_averaged_Psi_pixel_density_relative_error"],
        "verdict": (
            "EXISTING_ACTION_ABLATION_SITE_TRANSFER_PASS"
            if best["maximum_C19_averaged_Psi_pixel_density_relative_error"]
            < 0.05
            else "EXISTING_ACTION_ABLATIONS_SITE_TRANSFER_OPEN"),
        "claim_boundary": (
            "One plus-copy x-positive linear packet on side five. The test "
            "distinguishes existing action contributions; it does not fit "
            "or admit an action and does not close V4-006."),
    }
    (HERE / "discrete_site_transfer_action_ablation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
