"""Full tangent test of the minimal balanced-density substrate stiffness.

The positive reciprocal action is

  Ucarve = eta/(4 sum_w) sum_(i,n) w h_i h_(i+n)
           |chi_(i+n)-chi_i|^2,
  h_i = ((rhoChi_i-rhoPsi_i)/(rhoChi_i+rhoPsi_i))^2.

It is dimensionless, cubic, phase invariant, symmetric under exchange of the
two density arguments, equals one in either one-field vacuum, and has a smooth
double zero at equal densities.  No prepared ray, corner, track, or Dirac
object appears in it.  This script tests the exact complete Hessian; it does
not admit the action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, null_space
import torch

import canonical_constrained_index as canonical_index
import constrained_energy_index as reduced
import local_binding_branch_scan as binding
import localized_stability as native


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
WEIGHT_SUM = float(sum(weight for _, weight in native.DIRECTIONS))


def energy(vector: torch.Tensor, side: int):
    data = vector.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :]) / np.sqrt(native.B)
    # Keep second-order complex curvature explicit.  Nested differentiation
    # through ``abs(z)^2`` can silently lose complex Hessian entries.
    p = torch.sum(psi.real.square() + psi.imag.square(), dim=-1)
    c = torch.sum(chi.real.square() + chi.imag.square(), dim=-1)
    h = ((c - p) / (c + p))**2
    result = vector.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        shifts = tuple(-value for value in offset)
        neighbor_chi = torch.roll(chi, shifts, (0, 1, 2))
        neighbor_h = torch.roll(h, shifts, (0, 1, 2))
        difference = neighbor_chi - chi
        result = result + (weight / (4 * WEIGHT_SUM)
                           * torch.sum(h * neighbor_h
                                       * torch.sum(
                                           difference.real.square()
                                           + difference.imag.square(),
                                           dim=-1)))
    return result


def classify(side: int, lam: float, eta: float):
    amplitude_sq = native.R**2 / (2 + 54 / native.R**2)
    psi, chi = native.carrier_fields(side, amplitude_sq)
    fp, fc = binding.modified_force(psi, chi, lam, 1.0)
    wp2 = float(-np.vdot(psi, fp).real / np.vdot(psi, psi).real)
    omega = float(np.sqrt(wp2))
    configuration = native.pack(psi, chi)
    variable = torch.tensor(configuration, requires_grad=True)
    action_energy = energy(variable, side)
    gradient, = torch.autograd.grad(action_energy, variable, create_graph=True)
    hessian = torch.autograd.functional.hessian(
        lambda x: energy(x, side), variable,
        vectorize=True).detach().numpy()
    gradient = gradient.detach().numpy()
    force_jacobian = binding.modified_jacobian(psi, chi, lam, 1.0)
    gamma = canonical_index.rotation_generator(side, omega, 0.0)
    size = force_jacobian.shape[0]
    mass = np.eye(size)
    gyro = 2 * gamma
    stiffness = gamma @ gamma - force_jacobian + eta * hessian
    generator = np.block([
        [np.zeros_like(mass), np.eye(size)], [-stiffness, -gyro]])
    omega_form = np.block([
        [gyro, mass], [-mass, np.zeros_like(mass)]])
    augmented_hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass]])
    _, q_tangents = canonical_index.symmetry_directions(psi, chi)
    tangents = np.vstack((q_tangents, np.zeros_like(q_tangents)))
    tangents, _, _ = reduced.independent_columns(tangents)
    constraints, _, _ = reduced.independent_columns(
        np.column_stack((tangents, omega_form @ tangents)))
    basis = null_space(constraints.T, rcond=1e-10)
    _, index = reduced.index_counts(basis.T @ augmented_hessian @ basis)
    values = eig(generator, right=False, check_finite=False)
    return {
        "lambda_self": lam,
        "eta": eta,
        "amplitude_sq": float(amplitude_sq),
        "omega_psi": omega,
        "omega_chi": 0.0,
        "action_gradient_norm_at_orbit": float(np.linalg.norm(gradient)),
        "hessian_symmetry_defect": float(np.linalg.norm(hessian-hessian.T)
                                         / max(np.linalg.norm(hessian), 1e-30)),
        "maximum_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "constrained_hessian_index": index,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=(1.0, 0.25))
    parser.add_argument("--etas", nargs="+", type=float,
                        default=(0.0, 0.25, 1.0, 4.0, 16.0))
    parser.add_argument("--output", default="carved_substrate_tangent.json")
    args = parser.parse_args()
    native.B = 1197.0
    rows = [classify(args.side, lam, eta)
            for lam in args.lambdas for eta in args.etas]
    result = {
        "status": "V4_CARVED_SUBSTRATE_COMPLETE_TANGENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "claim_boundary": (
            "Exact-orbit complete tangent only; stability is required before "
            "nonlinear and movement gates."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
