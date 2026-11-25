"""Unrestricted constrained-energy solve for the minimal two-wave action.

The old eight-site field is only an initial condition and identity readout.
Every native field component is optimized.  Eliminating the single Psi phase
frequency at fixed Noether charge gives the standard energy--momentum
functional U + Q_Psi^2/(2 I_Psi).  Chi is static and carries zero charge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch

import localized_stability as native
import unrestricted_static_chi_profile as static_branch


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
WEIGHT_SUM = float(sum(weight for _, weight in native.DIRECTIONS))


def fields(vector: torch.Tensor, side: int):
    data = vector.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :]) / np.sqrt(native.B)
    return psi, chi


def potential(vector: torch.Tensor, side: int, lam: float, coupling: float,
              psi_gradient: float, chi_gradient: float):
    psi, chi = fields(vector, side)
    p = torch.sum(torch.abs(psi)**2, dim=-1)
    c = torch.sum(torch.abs(chi)**2, dim=-1)
    value = torch.sum((c-native.R**2)**2/4 + coupling*c*p/2
                      + lam*p**2/4)
    area = torch.linalg.cross(chi, psi, dim=-1)
    for offset, weight in native.DIRECTIONS:
        shifts = tuple(-x for x in offset)
        da = torch.roll(area, shifts, (0, 1, 2)) - area
        dp = torch.roll(psi, shifts, (0, 1, 2)) - psi
        dc = torch.roll(chi, shifts, (0, 1, 2)) - chi
        value = value + weight * (
            torch.sum(torch.abs(da)**2)/(2*native.R**2)
            + psi_gradient*torch.sum(torch.abs(dp)**2)/(4*WEIGHT_SUM)
            + chi_gradient*torch.sum(torch.abs(dc)**2)/(4*WEIGHT_SUM))
    return value


def objective(vector: torch.Tensor, side: int, lam: float, coupling: float,
              psi_gradient: float, chi_gradient: float, charge: float):
    psi, _ = fields(vector, side)
    inertia = torch.sum(torch.abs(psi)**2)
    return (potential(vector, side, lam, coupling,
                      psi_gradient, chi_gradient)
            + charge**2/(2*inertia))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--lambda-self", type=float, default=0.25)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--psi-gradient", type=float, default=1.0)
    parser.add_argument("--chi-gradient", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--output", default="fixed_charge_variational_profile.json")
    args = parser.parse_args()
    native.B = 1197.0
    amplitude_sq = native.R**2/(2+54/native.R**2)
    psi0, chi0 = native.carrier_fields(args.side, amplitude_sq)
    # Static-chi exact branch frequency for the chosen local action.
    rho = native.norm2(psi0)[..., None]
    canonical_fp, _ = native.force(psi0, chi0)
    fp = canonical_fp - (args.lambda_self-1)*rho*psi0
    omega0 = float(np.sqrt(
        -np.vdot(psi0, fp).real/np.vdot(psi0, psi0).real))
    charge = omega0*float(np.sum(abs(psi0)**2))
    initial = native.pack(psi0, chi0)
    if args.chi_gradient > 0:
        mask = native.norm2(psi0) > 0
        chi_seed = static_branch.harmonic_chi_seed(chi0, mask)
        initial = native.pack(psi0, chi_seed)

    evaluations = 0

    def value_gradient(x):
        nonlocal evaluations
        variable = torch.tensor(x, requires_grad=True)
        value = objective(variable, args.side, args.lambda_self,
                          args.coupling, args.psi_gradient,
                          args.chi_gradient, charge)
        gradient, = torch.autograd.grad(value, variable)
        evaluations += 1
        return float(value.detach()), gradient.detach().numpy()

    initial_value, initial_gradient = value_gradient(initial)
    solution = minimize(
        value_gradient, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": args.maxiter, "maxls": 50,
                 "ftol": 1e-14, "gtol": 1e-9, "maxcor": 30})
    psi, chi = native.unpack(solution.x, args.side)
    final_value, final_gradient = value_gradient(solution.x)
    mask = native.norm2(psi0) > 0
    psi_total = max(float(np.sum(abs(psi)**2)), 1e-30)
    overlap = (abs(np.vdot(psi0[mask], psi[mask]))**2
               / max(float(np.sum(abs(psi0[mask])**2)
                           * np.sum(abs(psi[mask])**2)), 1e-30))
    density = native.norm2(psi)
    participation = float(np.sum(density)**2
                          / max(np.sum(density**2), 1e-30))
    vacuum = np.zeros_like(chi)
    vacuum[..., 2] = native.R
    result = {
        "status": "V4_FIXED_CHARGE_VARIATIONAL_PROFILE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": args.side,
        "lambda_self": args.lambda_self,
        "density_coupling": args.coupling,
        "psi_gradient_action_coefficient": args.psi_gradient,
        "chi_gradient_action_coefficient": args.chi_gradient,
        "fixed_psi_charge": charge,
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "iterations": int(solution.nit),
        "evaluations": evaluations,
        "initial_augmented_energy": initial_value,
        "final_augmented_energy": final_value,
        "initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "psi_carrier_overlap_sq": float(overlap),
        "psi_exterior_fraction": float(np.sum(abs(psi[~mask])**2)/psi_total),
        "psi_participation_sites": participation,
        "chi_exterior_deviation_rms": float(np.linalg.norm(
            (chi-vacuum)[~mask])/np.sqrt(3*np.count_nonzero(~mask))),
        "claim_boundary": (
            "Constrained variational feasibility only. A localized minimum "
            "still requires Hessian, evolution, movement, and Dirac gates."),
    }
    output = HERE/args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), psi=psi, chi=chi,
                        parameters=solution.x)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
