"""Hamiltonian--Krein audit of the exact localized rest orbit.

Eigenvalue chasing is unreliable in a gyroscopic Hamiltonian system because
neutral modes can collide and leave the imaginary axis.  This audit rebuilds
the noncanonical symplectic form and quadratic Hamiltonian from the extracted
M, C, K pencil, verifies Hamiltonian structure, and records the energy
signature of every neutral mode and the zero-energy character of each
unstable quartet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import action_response_rank as rank
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--neutral-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", default="krein_audit.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = rank.exact_rest(args.rate)
    dimension = mass.shape[0]
    omega = np.block([
        [gyro, mass],
        [-mass, np.zeros_like(mass)],
    ])
    hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass],
    ])
    structure_residual = np.linalg.norm(
        generator.T @ omega + omega @ generator) / (
            np.linalg.norm(generator) * np.linalg.norm(omega))
    hessian_residual = np.linalg.norm(
        -omega @ generator - hessian) / max(np.linalg.norm(hessian), 1e-30)

    values, vectors = eig(generator, check_finite=False)
    rows = []
    positive = negative = zero = unstable = 0
    for index, value in enumerate(values):
        vector = vectors[:, index]
        vector /= np.linalg.norm(vector)
        energy = float(np.real(np.vdot(vector, hessian @ vector)))
        if abs(value.real) > args.neutral_tolerance:
            classification = "unstable_zero_energy_quartet"
            unstable += 1
        elif abs(energy) <= 1e-8:
            classification = "neutral_zero_signature"
            zero += 1
        elif energy > 0:
            classification = "neutral_positive_signature"
            positive += 1
        else:
            classification = "neutral_negative_signature"
            negative += 1
        rows.append({
            "real": float(value.real),
            "imag": float(value.imag),
            "quadratic_energy": energy,
            "classification": classification,
        })
    rows.sort(key=lambda row: (-abs(row["real"]), abs(row["imag"])))
    result = {
        "status": "V4_HAMILTONIAN_KREIN_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rate": args.rate,
        "dt": dt,
        "rest_del_rms_residual": residual,
        "hamiltonian_structure_relative_residual": float(structure_residual),
        "quadratic_hessian_reconstruction_residual": float(hessian_residual),
        "mode_counts": {
            "unstable_eigenvalues": unstable,
            "neutral_positive_signature": positive,
            "neutral_negative_signature": negative,
            "neutral_zero_signature": zero,
        },
        "modes": rows,
        "interpretation": (
            "Nonzero-real Hamiltonian quartets arise through collisions of "
            "opposite Krein signature. Candidate actions must repair the "
            "quadratic energy on the colliding subspace while preserving "
            "the phase and translation zero modes; fitting only existing "
            "positive growth rates is insufficient."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rest_del_rms_residual": residual,
        "hamiltonian_structure_relative_residual": structure_residual,
        "quadratic_hessian_reconstruction_residual": hessian_residual,
        "mode_counts": result["mode_counts"],
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
