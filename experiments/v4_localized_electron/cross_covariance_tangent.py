"""Tangent screen for a positive C19 wave of the full Psi--chi relation.

The native joint coordinate

    X_i = chi_i Psi_i^dagger

retains the direction and phase relation between every component of the two
complex fields.  Its normalized form Z=R X/sqrt(R^4+||X||_F^2) supplies the
single positive action

  eta/(2 sum_w) [<Zdot,D19 Zdot> - <Z,D19^2 Z>].

This file uses automatic differentiation only to evaluate the exact quadratic
action at the current research rest orbit.  It does not alter the native
equations.  A passing tangent screen would still require analytic reciprocal
equations, an unrestricted rest solve, and the full nonlinear gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
import torch

import action_response_rank as rank
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def unpack(raw: torch.Tensor, side: int):
    data = raw.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :])
    return psi, chi


def pack(psi: torch.Tensor, chi: torch.Tensor):
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def d19(value: torch.Tensor):
    result = torch.zeros_like(value)
    for offset, weight in field.prior.bond.local.common.base.DIRECTIONS:
        shifted = torch.roll(value, shifts=offset, dims=(0, 1, 2))
        result = result + weight * (value - shifted)
    return result


def rotation_velocity(raw: torch.Tensor, side: int,
                      omega_psi: float, omega_chi: float):
    psi, chi = unpack(raw, side)
    pf = torch.as_tensor((omega_psi, omega_psi, omega_psi - omega_chi),
                         dtype=raw.dtype, device=raw.device)
    cf = torch.as_tensor((omega_chi, omega_chi, 0.0),
                         dtype=raw.dtype, device=raw.device)
    return pack(1j * pf * psi, 1j * cf * chi)


def unit_lagrangian_parts(state: torch.Tensor, side: int,
                          omega_psi: float, omega_chi: float):
    dimension = state.numel() // 2
    raw = state[:dimension]
    co_velocity = state[dimension:]
    physical_velocity = co_velocity + rotation_velocity(
        raw, side, omega_psi, omega_chi)
    psi, chi = unpack(raw, side)
    vp, vc = unpack(physical_velocity, side)
    x = chi[..., :, None] * psi.conj()[..., None, :]
    xdot = (vc[..., :, None] * psi.conj()[..., None, :]
            + chi[..., :, None] * vp.conj()[..., None, :])
    density = torch.sum(abs(x)**2, dim=(-2, -1), keepdim=True)
    denominator = field.prior.R**4 + density
    h = field.prior.R / torch.sqrt(denominator)
    hprime = -field.prior.R / (2 * denominator**1.5)
    radial = torch.real(torch.sum(x.conj() * xdot,
                                  dim=(-2, -1), keepdim=True))
    z = h * x
    zdot = h * xdot + 2 * hprime * radial * x
    dz = d19(z)
    kinetic = torch.real(torch.sum(zdot.conj() * d19(zdot)))
    potential = torch.real(torch.sum(dz.conj() * dz))
    scale = 2 * field.prior.bond.C19_WEIGHT_SUM
    return kinetic / scale, -potential / scale


def unit_bond_potential(state: torch.Tensor, side: int):
    """Negative of the positive one-hop C19 normalized-X bond energy."""
    dimension = state.numel() // 2
    psi, chi = unpack(state[:dimension], side)
    x = chi[..., :, None] * psi.conj()[..., None, :]
    density = torch.sum(abs(x)**2, dim=(-2, -1), keepdim=True)
    z = field.prior.R * x / torch.sqrt(field.prior.R**4 + density)
    potential = torch.real(torch.sum(z.conj() * d19(z)))
    return -potential / (2 * field.prior.bond.C19_WEIGHT_SUM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--etas", nargs="+", type=float,
                        default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--output", default="cross_covariance_tangent.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = rank.exact_rest(args.rate)
    q = torch.tensor(layers[1], dtype=torch.float64)
    state = torch.cat((q, torch.zeros_like(q))).requires_grad_(True)
    omega_psi = float(solution.x[1] / dt)
    omega_chi = float(solution.x[2] / dt)
    gradients = []
    hessians = []
    for part in range(2):
        function = lambda value, part=part: unit_lagrangian_parts(
            value, side, omega_psi, omega_chi)[part]
        gradients.append(torch.autograd.functional.jacobian(
            function, state, vectorize=True).detach().numpy())
        hessians.append(torch.autograd.functional.hessian(
            function, state, vectorize=True).detach().numpy())
    bond_gradient = torch.autograd.functional.jacobian(
        lambda value: unit_bond_potential(value, side), state,
        vectorize=True).detach().numpy()
    bond_hessian = torch.autograd.functional.hessian(
        lambda value: unit_bond_potential(value, side), state,
        vectorize=True).detach().numpy()
    dimension = mass.shape[0]
    qgrad_k = gradients[0][:dimension]
    qgrad_v = gradients[1][:dimension]
    speed_sq = float(-np.dot(qgrad_k, qgrad_v)
                     / max(np.dot(qgrad_v, qgrad_v), 1e-30))
    gradient = gradients[0] + speed_sq * gradients[1]
    hessian = hessians[0] + speed_sq * hessians[1]

    def blocks(source):
        dqq = source[:dimension, :dimension]
        dqv = source[:dimension, dimension:]
        dvq = source[dimension:, :dimension]
        dvv = source[dimension:, dimension:]
        return ((dvv + dvv.T) / 2, dvq - dqv,
                -(dqq + dqq.T) / 2)

    dmass, dgyro, dstiffness = blocks(hessian)

    def spectral_row(label, eta, increments):
        imass, igyro, istiffness = increments
        trial_mass = mass + eta * imass
        trial_gyro = gyro + eta * igyro
        trial_stiffness = stiffness + eta * istiffness
        mass_values = np.linalg.eigvalsh((trial_mass + trial_mass.T) / 2)
        size = trial_mass.shape[0]
        trial_generator = np.block([
            [np.zeros_like(trial_mass), np.eye(size)],
            [-np.linalg.solve(trial_mass, trial_stiffness),
             -np.linalg.solve(trial_mass, trial_gyro)],
        ])
        values = eig(trial_generator, right=False, check_finite=False)
        return {
            "component": label,
            "eta": eta,
            "maximum_growth": float(np.max(values.real)),
            "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
            "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
            "mass_minimum_eigenvalue": float(mass_values[0]),
            "mass_negative_count": int(np.sum(mass_values < -1e-8)),
        }
    rows = []
    for eta in [0.0] + args.etas:
        rows.append(spectral_row("rest_ratio", eta,
                                 (dmass, dgyro, dstiffness)))
    component_rows = []
    for label, source in (("kinetic_only", hessians[0]),
                          ("potential_only", hessians[1])):
        increments = blocks(source)
        for eta in (1e-8, 1e-7, 1e-6, 1e-5):
            component_rows.append(spectral_row(label, eta, increments))
    bond_blocks = blocks(bond_hessian)
    for eta in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        component_rows.append(spectral_row(
            "one_hop_bond_potential", eta, bond_blocks))
    result = {
        "status": "V4_CROSS_COVARIANCE_TANGENT_RESEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rest_del_rms_residual": residual,
        "rest_derived_speed_squared": speed_sq,
        "kinetic_rest_gradient_norm": float(np.linalg.norm(qgrad_k)),
        "potential_rest_gradient_norm": float(np.linalg.norm(qgrad_v)),
        "bond_potential_rest_gradient_norm": float(np.linalg.norm(
            bond_gradient[:dimension])),
        "kinetic_potential_gradient_cosine": float(
            np.dot(qgrad_k, qgrad_v)
            / max(np.linalg.norm(qgrad_k) * np.linalg.norm(qgrad_v), 1e-30)),
        "unit_action_rest_gradient_norm": float(np.linalg.norm(
            gradient[:dimension])),
        "dmass_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(dmass))),
        "dmass_maximum_eigenvalue": float(np.max(np.linalg.eigvalsh(dmass))),
        "symmetry_checks": {
            "mass_asymmetry": float(np.linalg.norm(dmass - dmass.T)),
            "gyro_symmetric_part": float(np.linalg.norm(dgyro + dgyro.T)),
            "stiffness_asymmetry": float(np.linalg.norm(
                dstiffness - dstiffness.T)),
        },
        "rows": rows,
        "component_rows": component_rows,
        "claim_boundary": (
            "Fixed-orbit quadratic screen only. A favorable row requires "
            "analytic variation and a freely re-solved nonlinear orbit."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
