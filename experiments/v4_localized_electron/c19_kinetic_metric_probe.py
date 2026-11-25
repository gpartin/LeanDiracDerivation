"""Test the smallest positive C19-neighbor chi kinetic metric.

The candidate replaces the local chi kinetic term by

  T_chi = B/2 [kappa ||chi_dot||^2 + ||A19 chi_dot / 24||^2],

where ``A19`` is the weighted C19 adjacency (six face weights two and
twelve edge weights one).  Since ``D19 = 24 I - A19``, this is the positive
regularization of the response oracle ``(I-D19/24)^2``.  It contains no
carrier, corner, direction, copy, or Dirac data.  The identity
``B*kappa=chi0=19`` fixes the local inertia without fitting movement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE_MOVEMENT = 5
SIDE_SPECTRUM = 3
DT_SPECTRUM = 1.0 / 256.0


def adjacency_average(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        result += weight * rank.native.shift(value, offset)
    return result / rank.C19_WEIGHT_SUM


def metric(value: np.ndarray, power: int = 1) -> np.ndarray:
    result = value
    for _ in range(2 * power):
        result = adjacency_average(result)
    return rank.KAPPA * value + result


def inverse_metric(value: np.ndarray, power: int = 1) -> np.ndarray:
    """Apply [kappa I + (A19/24)^(2 power)]^-1 by periodic FFT."""
    side = value.shape[0]
    transform = np.fft.fftn(value, axes=(0, 1, 2))
    frequencies = 2 * np.pi * np.fft.fftfreq(side)
    denominator = np.empty((side, side, side))
    for ix, kx in enumerate(frequencies):
        for iy, ky in enumerate(frequencies):
            for iz, kz in enumerate(frequencies):
                cosine = np.cos((kx, ky, kz))
                d19 = 24 - 4 * (
                    np.sum(cosine)
                    + cosine[0] * cosine[1]
                    + cosine[0] * cosine[2]
                    + cosine[1] * cosine[2])
                adjacency_symbol = 1 - d19 / rank.C19_WEIGHT_SUM
                denominator[ix, iy, iz] = (
                    rank.KAPPA + adjacency_symbol ** (2 * power))
    return np.fft.ifftn(
        transform / denominator[..., None], axes=(0, 1, 2))


def movement_rows(power: int):
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    rows = evidence["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE_MOVEMENT, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    output = []
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        velocity_minus_psi = (layers[1][0] - layers[0][0]) / dt
        velocity_plus_psi = (layers[2][0] - layers[1][0]) / dt
        velocity_minus_chi = (layers[1][1] - layers[0][1]) / dt
        velocity_plus_chi = (layers[2][1] - layers[1][1]) / dt
        residual = rank.pack(
            velocity_minus_psi - velocity_plus_psi + dt * force_psi,
            rank.B * (
                metric(velocity_minus_chi, power)
                - metric(velocity_plus_chi, power)
                + dt * force_chi),
        )
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        numerator += float(np.vdot(residual, residual).real)
        denominator += float(np.vdot(baseline, baseline).real)
        output.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(
                np.linalg.norm(residual) / np.linalg.norm(baseline)),
        })
    return float(np.sqrt(numerator / denominator)), output


def apply_inverse_to_packed(vector: np.ndarray, side: int, power: int):
    psi, chi = rank.native.unpack(vector, side)
    return rank.native.pack(psi, inverse_metric(chi, power))


def transformed_tangent(matrix: np.ndarray, side: int, power: int):
    return np.column_stack([
        apply_inverse_to_packed(matrix[:, column], side, power)
        for column in range(matrix.shape[1])
    ])


def rest_spectrum(power: int):
    side = SIDE_SPECTRUM
    dt = DT_SPECTRUM
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(side, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    theta = float(np.arccos(1 - dt**2 * rank.R**2 / 2))
    qp = np.exp(1j * theta)
    qc = 1.0 + 0.0j
    pieces = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    canonical_tangent = pieces[1] + pieces[2]
    tangent = (canonical_tangent if power == 0 else
               transformed_tangent(canonical_tangent, side, power))
    rotation_minus = rank.native.rotation_matrix(side, qp, qc, -1)
    rotation_minus_two = rank.native.rotation_matrix(side, qp, qc, -2)
    identity = np.eye(tangent.shape[0])
    zero = np.zeros_like(identity)
    monodromy = np.block([
        [rotation_minus @ (2 * identity + dt**2 * tangent),
         -rotation_minus_two],
        [identity, zero],
    ])
    values = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(values))
    radius = float(moduli[-1])
    return {
        "power": power,
        "orbit_force_defect_psi": float(
            np.linalg.norm(force_psi + rank.R**2 * psi)),
        "orbit_force_defect_chi": float(np.linalg.norm(force_chi)),
        "spectral_radius": radius,
        "growth_rate_per_unit_time": float(np.log(radius) / dt),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
    }


def main() -> None:
    rank.lock_inertia()
    movement = []
    for power in range(1, 5):
        combined, rows = movement_rows(power)
        movement.append({
            "power": power,
            "combined_relative_residual": combined,
            "rows": rows,
        })
    spectra = [rest_spectrum(power) for power in range(0, 4)]
    baseline_growth = spectra[0]["growth_rate_per_unit_time"]
    candidate_growth = spectra[1]["growth_rate_per_unit_time"]
    result = {
        "status": "V4_C19_KINETIC_METRIC_PARTIAL_IMPROVEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "T_chi=(B/2)[kappa ||chiDot||^2+"
            "||A19 chiDot/24||^2]"),
        "B": rank.B,
        "kappa": rank.KAPPA,
        "B_times_kappa": rank.B * rank.KAPPA,
        "chi0": rank.R,
        "coefficient_provenance": (
            "24 is the weighted C19 degree; B*kappa=chi0=19; no movement "
            "or Dirac target fixes a coefficient"),
        "kinetic_symbol": "B[kappa+(A19(k)/24)^2]",
        "symbol_bounds": [rank.B * rank.KAPPA,
                          rank.B * (1 + rank.KAPPA)],
        "positive_and_invertible": True,
        "rest_orbit_unchanged": True,
        "movement_controls": movement,
        "rest_spectrum_controls": spectra,
        "power_one_growth_relative_improvement": float(
            (baseline_growth - candidate_growth) / baseline_growth),
        "verdict": "PARTIAL_IMPROVEMENT_NOT_MOVEMENT_PASS",
        "root_cause": (
            "The canonical local B inertia over-penalizes the nonuniform chi "
            "motion required by a one-edge handoff.  The C19-average metric "
            "reduces that resistance with native coefficients and improves "
            "the rest spectrum, but it supplies no field-dependent connection "
            "force, so the exact translated recurrence remains nonzero."),
        "next_action": (
            "Keep only power one as the locality-minimal incumbent.  Derive "
            "a positive phase-covariant field-dependent connection inside "
            "the same square, and reject it unless it improves both the "
            "0.10687 movement residual and the 0.75167 rest growth."),
        "claim_boundary": (
            "The calculation is a side-5 fixed-phase movement response and "
            "a side-3 linear rest spectrum.  It is not a freely translated "
            "orbit, long-time stability result, Dirac pass, or admitted action."),
    }
    (HERE / "c19_kinetic_metric_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
