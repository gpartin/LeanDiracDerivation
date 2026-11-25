"""Solve the nearby native charge/frequency family of the C27/gyro carrier.

Arbitrary previous-layer kicks are not neighboring stationary particles.  For
each prescribed Psi clock or internal chi clock, this script freely re-solves
the independent carrier Psi and chi amplitudes under the *same fixed action*.
It then derives the two native Noether charges and centered two-layer family
tangents.  These tangents are the lawful collective quotient for perturbation
and movement tests; no runtime projection is introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch

import c27_body_gyro_basis_probe as candidate
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
FRACTION = 0.01


def carrier_fields(psi_amplitude_sq: float, chi_amplitude_sq: float):
    unit_psi, unit_chi = candidate.base.inverse.rank.native.carrier_fields(
        candidate.base.inverse.SIDE, 1.0)
    occupied = np.sum(abs(unit_psi)**2, axis=-1) > 1e-12
    psi = np.sqrt(psi_amplitude_sq) * unit_psi
    chi = unit_chi.copy()
    chi[occupied] = np.sqrt(chi_amplitude_sq) * unit_chi[occupied]
    return psi, chi


def layers(parameters: np.ndarray, theta_psi: float, theta_chi: float):
    psi, chi = carrier_fields(np.exp(parameters[0]), np.exp(parameters[1]))
    previous = relative.rotate(psi, chi, theta_psi, theta_chi, -1)
    following = relative.rotate(psi, chi, theta_psi, theta_chi, 1)
    return (
        candidate.base.inverse.rank.pack(*previous),
        candidate.base.inverse.rank.pack(psi, chi),
        candidate.base.inverse.rank.pack(*following),
    )


def phase_generator(raw: np.ndarray, field: str):
    psi, chi = relative.unpack_raw(raw, candidate.base.inverse.SIDE)
    generator_psi = np.zeros_like(psi)
    generator_chi = np.zeros_like(chi)
    if field == "psi":
        generator_psi = 1j * psi
    elif field == "chi_internal":
        generator_psi[..., 2] = -1j * psi[..., 2]
        generator_chi[..., :2] = 1j * chi[..., :2]
    else:
        raise ValueError(field)
    return relative.pack_raw(generator_psi, generator_chi)


def main() -> None:
    candidate.base.inverse.rank.lock_inertia()
    candidate.base.inverse.rest.SIDE = candidate.base.inverse.SIDE
    candidate.base.inverse.rest.DT = candidate.base.inverse.DT
    previous0, current0, following0, theta0 = (
        candidate.base.inverse.rest_layers())
    dt = candidate.base.inverse.DT
    a = 2 * dt / np.sin(theta0)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    del_residual, right_momentum = candidate.make_derivatives(
        1.0, coefficients)
    amplitude0 = candidate.base.inverse.rank.R**2 / (
        2 + 54 / candidate.base.inverse.rank.R**2)
    initial = np.log((amplitude0, amplitude0))
    scale = np.sqrt(previous0.size)

    cases = (
        ("baseline", theta0, 0.0),
        ("psi_clock_minus", theta0 * (1 - FRACTION), 0.0),
        ("psi_clock_plus", theta0 * (1 + FRACTION), 0.0),
        ("chi_clock_minus", theta0, -theta0 * FRACTION),
        ("chi_clock_plus", theta0, theta0 * FRACTION),
    )
    solved = {}
    rows = []
    for name, theta_psi, theta_chi in cases:
        def residual(parameters):
            state = tuple(torch.tensor(value, dtype=torch.float64)
                          for value in layers(parameters, theta_psi, theta_chi))
            return del_residual(*state).detach().numpy() / scale

        solution = least_squares(
            residual, initial.copy(), xtol=1e-13, ftol=1e-13, gtol=1e-13,
            max_nfev=200)
        state = layers(solution.x, theta_psi, theta_chi)
        tensors = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in state)
        raw_residual = del_residual(*tensors).detach().numpy()
        momentum = right_momentum(tensors[0], tensors[1]).detach().numpy()
        charges = {
            field: float(np.dot(momentum, phase_generator(state[1], field)))
            for field in ("psi", "chi_internal")
        }
        solved[name] = state
        rows.append({
            "name": name,
            "theta_psi": theta_psi,
            "theta_chi": theta_chi,
            "omega_psi": theta_psi / dt,
            "omega_chi": theta_chi / dt,
            "psi_amplitude_sq": float(np.exp(solution.x[0])),
            "chi_amplitude_sq": float(np.exp(solution.x[1])),
            "rest_DEL_rms": float(
                np.linalg.norm(raw_residual) / np.sqrt(raw_residual.size)),
            "solver_success": bool(solution.success),
            "solver_evaluations": int(solution.nfev),
            "noether_charges": charges,
        })

    rotation = relative.raw_rotation_matrix(
        candidate.base.inverse.SIDE, theta0, 0.0, 1)
    phase_states = {}
    for name, state in solved.items():
        phase_states[name] = np.r_[state[1], rotation @ state[0]]
    psi_tangent = (
        phase_states["psi_clock_plus"] - phase_states["psi_clock_minus"]
    ) / (2 * FRACTION)
    chi_tangent = (
        phase_states["chi_clock_plus"] - phase_states["chi_clock_minus"]
    ) / (2 * FRACTION)
    tangent_overlap = float(abs(np.dot(psi_tangent, chi_tangent)) / max(
        np.linalg.norm(psi_tangent) * np.linalg.norm(chi_tangent), 1e-300))
    maximum_residual = max(row["rest_DEL_rms"] for row in rows)
    family_pass = maximum_residual < 1e-8
    np.savez_compressed(
        HERE / "c27_body_gyro_charge_family.npz",
        baseline=phase_states["baseline"],
        psi_clock_tangent=psi_tangent,
        chi_clock_tangent=chi_tangent,
    )
    result = {
        "status": "V4_C27_BODY_GYRO_CHARGE_FAMILY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": candidate.base.inverse.SIDE,
        "dt": dt,
        "fraction": FRACTION,
        "fixed_action_coefficients": {
            name: float(value)
            for name, value in zip(candidate.BASIS_NAMES, coefficients)
        },
        "rows": rows,
        "maximum_rest_DEL_rms": maximum_residual,
        "psi_chi_family_tangent_overlap": tangent_overlap,
        "family_tangents_linearly_independent": tangent_overlap < 1 - 1e-8,
        "verdict": (
            "C27_BODY_GYRO_NATIVE_CHARGE_FAMILY_PASS" if family_pass else
            "C27_BODY_GYRO_NATIVE_CHARGE_FAMILY_NOT_CLOSED"),
        "claim_boundary": (
            "Nearby relative-equilibrium family only. It supplies lawful "
            "collective perturbation tangents but does not prove nonlinear "
            "stability, translation, or Dirac-current matching."),
    }
    output = HERE / "c27_body_gyro_charge_family.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "maximum_rest_DEL_rms": maximum_residual,
        "tangent_overlap": tangent_overlap,
        "rows": [{
            "name": row["name"],
            "psi_amplitude_sq": row["psi_amplitude_sq"],
            "chi_amplitude_sq": row["chi_amplitude_sq"],
            "rest_DEL_rms": row["rest_DEL_rms"],
        } for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
