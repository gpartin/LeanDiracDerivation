"""Audit whether the latest componentwise material connection contains light.

This audit is deliberately scoped to the newest multichannel-prong candidate.
It does not test or modify the canonical action.  It asks whether the local
best-matching velocity and its near-null directions themselves provide a
linear, massless, transverse wave sector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as candidate


HERE = Path(__file__).resolve().parent
SIDE = candidate.SIDE


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.real**2)).detach())


def field_tests(epsilon: float) -> dict[str, float]:
    coordinates = torch.arange(SIDE, dtype=torch.float64)
    x = coordinates[:, None, None].expand(SIDE, SIDE, SIDE)
    wave_number = 2 * np.pi / SIDE
    omega = wave_number

    chi = torch.zeros((SIDE, SIDE, SIDE, 3), dtype=torch.complex128)
    chi[..., 2] = rank.R
    chi_dot = torch.zeros_like(chi)

    # A single transverse complex phasor.  Its time derivative is a local
    # phase rotation and is therefore intentionally removed by the candidate.
    phasor = torch.exp(1j * wave_number * x)
    phase_wave = torch.zeros_like(chi)
    phase_wave[..., 1] = epsilon * phasor
    phase_dot = -1j * omega * phase_wave
    phase_velocity = candidate.best_matching_velocity(
        phase_wave, phase_dot, chi, chi_dot, "psi")

    # A translating real envelope.  This is the kind of motion the material
    # connection was designed to infer.
    envelope = torch.zeros_like(chi)
    envelope[..., 1] = epsilon * (1 + 0.25 * torch.cos(wave_number * x))
    envelope_dot = torch.zeros_like(chi)
    envelope_dot[..., 1] = (
        epsilon * 0.25 * omega * torch.sin(wave_number * x))
    envelope_velocity = candidate.best_matching_velocity(
        envelope, envelope_dot, chi, chi_dot, "psi")
    advected = candidate.advect_same_components(
        envelope, envelope_velocity)
    base_kinetic = torch.sum(torch.abs(envelope_dot) ** 2) / 2
    material_kinetic = torch.sum(
        torch.abs(envelope_dot + advected) ** 2) / 2

    return {
        "epsilon": epsilon,
        "phase_wave_velocity_rms": rms(phase_velocity),
        "envelope_velocity_rms": rms(envelope_velocity),
        "envelope_material_kinetic_delta": float(
            (material_kinetic - base_kinetic).detach()),
    }


def carrier_metric() -> dict[str, object]:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi_np, _ = rank.native.carrier_fields(SIDE, amplitude_sq)
    psi = torch.tensor(psi_np, dtype=torch.complex128)
    gram, _ = candidate.normal_blocks(psi, torch.zeros_like(psi), 1.0)
    eigenvalues = torch.linalg.eigvalsh(gram).detach().numpy().reshape(-1, 3)
    traces = eigenvalues.sum(axis=1)
    active = eigenvalues[traces > 1e-12]
    ratios = active[:, 0] / np.maximum(active[:, -1], 1e-300)
    return {
        "active_sites": int(active.shape[0]),
        "minimum_eigenvalue": float(active.min()),
        "maximum_eigenvalue": float(active.max()),
        "median_smallest_to_largest_ratio": float(np.median(ratios)),
        "minimum_smallest_to_largest_ratio": float(ratios.min()),
        "interpretation": (
            "These eigenvalues measure local translation observability.  "
            "They are not frequencies of propagating normal modes."
        ),
    }


def main() -> None:
    rows = [field_tests(value) for value in
            (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]
    epsilon = np.asarray([row["epsilon"] for row in rows])
    velocity = np.asarray([row["envelope_velocity_rms"] for row in rows])
    kinetic = np.abs(np.asarray([
        row["envelope_material_kinetic_delta"] for row in rows]))
    # The small-amplitude half of the scan isolates the fixed regularizer.
    velocity_power = float(np.polyfit(
        np.log(epsilon[:4]), np.log(velocity[:4]), 1)[0])
    kinetic_power = float(np.polyfit(
        np.log(epsilon[:4]), np.log(kinetic[:4]), 1)[0])

    output = {
        "scope": "latest componentwise Psi/chi material connection only",
        "field_rows": rows,
        "small_amplitude_velocity_power": velocity_power,
        "small_amplitude_kinetic_correction_power": kinetic_power,
        "carrier_local_metric": carrier_metric(),
        "analytic_order": {
            "gram": "O(epsilon^2)",
            "rhs": "O(epsilon^2)",
            "velocity_with_fixed_vacuum_floor": "O(epsilon^2)",
            "advective_derivative": "O(epsilon^3)",
            "action_correction": "O(epsilon^4)",
        },
        "verdict": "LATEST_PRONG_CONNECTION_IS_NOT_YET_A_LIGHT_SECTOR",
        "reason": (
            "The candidate removes a pure local phase rate, has no linear "
            "vacuum contribution, and contains no plaquette-curvature "
            "dynamics.  Its near-null directions are under-observed local "
            "translations, not demonstrated transverse photon modes."
        ),
        "next_falsifiable_gate": (
            "Derive a bond connection from relative Psi/chi prong phase and "
            "orientation, form its cube-plaquette holonomy without adding a "
            "third field, and test for exactly two gapless transverse modes "
            "plus a nonpropagating gradient null direction."
        ),
        "claim_boundary": (
            "This does not show that the full two-field framework cannot "
            "support light.  It shows only that the newest best-matching "
            "material connection does not by itself supply linear light."
        ),
    }
    path = HERE / "latest_prong_light_sector_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
