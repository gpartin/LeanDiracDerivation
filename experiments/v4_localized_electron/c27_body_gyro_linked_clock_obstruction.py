"""Localize the first nonlinear obstruction to the linked clock family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent


def real_norm_sq(value: np.ndarray) -> float:
    return float(np.sum(abs(value) ** 2))


def field_channels(state: np.ndarray, residual: np.ndarray) -> dict:
    denominator = np.sum(abs(state) ** 2, axis=-1)
    occupied = denominator > 1e-18
    radial = np.zeros_like(state)
    phase = np.zeros_like(state)
    radial_coefficient = np.zeros(denominator.shape)
    phase_coefficient = np.zeros(denominator.shape)
    radial_coefficient[occupied] = np.real(np.sum(
        np.conj(state[occupied]) * residual[occupied], axis=-1
    )) / denominator[occupied]
    phase_coefficient[occupied] = np.real(np.sum(
        np.conj(1j * state[occupied]) * residual[occupied], axis=-1
    )) / denominator[occupied]
    radial[occupied] = radial_coefficient[occupied, None] * state[occupied]
    phase[occupied] = phase_coefficient[occupied, None] * 1j * state[occupied]
    transverse = residual - radial - phase
    total = max(real_norm_sq(residual), 1e-300)
    return {
        "radial_fraction": real_norm_sq(radial) / total,
        "phase_fraction": real_norm_sq(phase) / total,
        "transverse_fraction": real_norm_sq(transverse) / total,
    }


def summarize(label: str, state_raw: np.ndarray,
              residual_raw: np.ndarray, carrier: np.ndarray) -> dict:
    psi, chi = relative.unpack_raw(state_raw, spectrum.SIDE)
    rpsi, rchi = relative.unpack_raw(residual_raw, spectrum.SIDE)
    total = max(real_norm_sq(rpsi) + real_norm_sq(rchi), 1e-300)
    coords = list(np.ndindex((spectrum.SIDE,) * 3))
    parity = np.asarray([sum(coord) % 2 for coord in coords]).reshape(
        (spectrum.SIDE,) * 3)

    def field_summary(state, residual):
        site_norm = np.sum(abs(residual) ** 2, axis=-1)
        field_total = max(float(np.sum(site_norm)), 1e-300)
        flat_order = np.argsort(site_norm.reshape(-1))[::-1][:12]
        top = []
        for flat in flat_order:
            coord = tuple(int(value) for value in np.unravel_index(
                int(flat), site_norm.shape))
            top.append({
                "site": list(coord),
                "carrier": bool(carrier[coord]),
                "parity": int(parity[coord]),
                "fraction_of_field_residual": float(
                    site_norm[coord] / field_total),
                "component_norms": [float(value) for value in
                                    abs(residual[coord])],
            })
        return {
            "fraction_of_total_residual": field_total / total,
            "carrier_fraction": float(np.sum(site_norm[carrier]) / field_total),
            "exterior_fraction": float(np.sum(site_norm[~carrier]) / field_total),
            "even_parity_fraction": float(np.sum(site_norm[parity == 0])
                                           / field_total),
            "odd_parity_fraction": float(np.sum(site_norm[parity == 1])
                                          / field_total),
            "component_fractions": [float(value / field_total) for value in
                                    np.sum(abs(residual) ** 2,
                                           axis=(0, 1, 2))],
            "local_channels": field_channels(state, residual),
            "top_sites": top,
        }

    return {
        "label": label,
        "DEL_rms": float(np.linalg.norm(residual_raw)
                         / np.sqrt(residual_raw.size)),
        "psi": field_summary(psi, rpsi),
        "chi": field_summary(chi, rchi),
    }


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, theta0 = candidate.base.inverse.rest_layers()
    psi0, _ = relative.unpack_raw(current, spectrum.SIDE)
    carrier = np.sum(abs(psi0) ** 2, axis=-1) > 1e-12
    a = 2 * spectrum.DT / np.sin(theta0)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    residual, _ = candidate.make_derivatives(1.0, coefficients)
    data = np.load(HERE / "c27_body_gyro_linked_clock_continuation.npz")
    rows = []
    raw_residuals = []
    for prefix in ("minus", "plus"):
        previous = data[f"{prefix}_previous"]
        state = data[f"{prefix}_state"]
        following = data[f"{prefix}_following"]
        tensors = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in (previous, state, following))
        value = residual(*tensors).detach().numpy()
        raw_residuals.append(value)
        rows.append(summarize(prefix, state, value, carrier))
    cosine = float(np.dot(raw_residuals[0], raw_residuals[1]) / max(
        np.linalg.norm(raw_residuals[0]) * np.linalg.norm(raw_residuals[1]),
        1e-300))
    result = {
        "status": "V4_C27_BODY_GYRO_LINKED_CLOCK_OBSTRUCTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "opposite_branch_residual_cosine": cosine,
        "verdict": "C27_BODY_GYRO_NONLINEAR_OBSTRUCTION_LOCALIZED",
        "claim_boundary": (
            "RCA of the first finite linked-clock obstruction only. It does "
            "not establish stability, translation, or Dirac closure."),
    }
    output = HERE / "c27_body_gyro_linked_clock_obstruction.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
