"""Solve the symmetry-reduced nearby orbit at nonzero chi phase charge.

The exact carrier has only two density classes: eight carrier sites with
equal Psi/chi density and ordinary vacuum with Psi=0, |chi|^2=chi0^2.  This
script solves the three corresponding radial amplitudes for prescribed
discrete phase advances and then checks the unrestricted DEL residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch

import personal_touch_unit_geometric_nonlinear as nonlinear
import canonical_noncoplanar_bifurcation_audit as small


HERE = Path(__file__).resolve().parent


def main() -> None:
    nonlinear.ACTION_KIND = "reduced_interface"
    raw0 = np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"]
    reference = torch.tensor(raw0, dtype=torch.float64)
    psi0, chi0 = nonlinear.unpack(reference)
    carrier = nonlinear.norm2(psi0) > 1e-12
    vacuum = ~carrier
    dt = small.DT
    theta_psi = float(np.arccos(1-dt**2*small.CHI0**2/2))
    theta_chi = 0.01*theta_psi
    lambda_psi = 4*np.sin(theta_psi/2)**2/dt**2
    lambda_chi = 4*np.sin(theta_chi/2)**2/dt**2
    basis = []
    basis.append(nonlinear.pack(psi0, torch.zeros_like(chi0)))
    basis.append(nonlinear.pack(
        torch.zeros_like(psi0), chi0*carrier[..., None]))
    basis.append(nonlinear.pack(
        torch.zeros_like(psi0), chi0*vacuum[..., None]))
    basis_matrix = torch.stack(basis)
    basis_norm2 = torch.sum(basis_matrix*basis_matrix, dim=1)
    mass = nonlinear.masses(torch.device("cpu"))

    def state(scales: np.ndarray) -> torch.Tensor:
        psi = scales[0]*psi0
        chi = (scales[1]*carrier[..., None]
               + scales[2]*vacuum[..., None])*chi0
        return nonlinear.pack(psi, chi)

    def residual(scales: np.ndarray) -> torch.Tensor:
        point = state(scales).detach().requires_grad_(True)
        gradient = torch.func.grad(nonlinear.total_potential)(point)
        psi, chi = nonlinear.unpack(point)
        target = mass*nonlinear.pack(lambda_psi*psi, lambda_chi*chi)
        return gradient.detach()-target.detach()

    equation_scale = small.CHI0**3

    def equations(scales: np.ndarray) -> np.ndarray:
        value = residual(scales)
        return np.array([
            float(torch.dot(value, vector)/norm/equation_scale)
            for vector, norm in zip(basis_matrix, basis_norm2, strict=True)])

    solved = least_squares(
        equations, np.ones(3), bounds=(0.1, 3.0),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=500)
    final = residual(solved.x)
    projected = torch.sum(
        (torch.mv(basis_matrix, final)**2)/basis_norm2)
    total = torch.sum(final*final)
    orthogonal = torch.sqrt(torch.clamp(total-projected, min=0))
    final_state = state(solved.x)
    psi, chi = nonlinear.unpack(final_state)
    result = {
        "status": "V4_REDUCED_INTERFACE_CHARGED_RELATIVE_ORBIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "chi_clock_fraction": 0.01,
        "theta_Psi": theta_psi,
        "theta_chi": theta_chi,
        "lambda_Psi": lambda_psi,
        "lambda_chi": lambda_chi,
        "solver_success": bool(solved.success),
        "solver_message": str(solved.message),
        "solver_cost": float(solved.cost),
        "scales": {
            "Psi_carrier": float(solved.x[0]),
            "chi_carrier": float(solved.x[1]),
            "chi_vacuum": float(solved.x[2]),
        },
        "normalized_projected_equations": equations(solved.x).tolist(),
        "full_DEL_residual_norm": float(torch.linalg.vector_norm(final)),
        "orthogonal_DEL_residual_norm": float(orthogonal),
        "relative_orthogonal_fraction": float(
            orthogonal/max(torch.linalg.vector_norm(final),
                           torch.tensor(1e-300))),
        "densities": {
            "Psi_carrier": float(torch.mean(nonlinear.norm2(psi)[carrier])),
            "chi_carrier": float(torch.mean(nonlinear.norm2(chi)[carrier])),
            "chi_vacuum": float(torch.mean(nonlinear.norm2(chi)[vacuum])),
        },
        "verdict": (
            "SYMMETRY_REDUCED_CHARGED_BRANCH_PASS"
            if solved.success
            and float(torch.linalg.vector_norm(final)) < 1e-8
            else "SYMMETRY_REDUCED_CHARGED_BRANCH_INSUFFICIENT"),
        "claim_boundary": (
            "Three-amplitude symmetry solve for a one-percent chi phase "
            "advance. A pass establishes the nearby relative orbit; it does "
            "not establish perturbative stability or movement."),
    }
    (HERE/"personal_touch_reduced_interface_charged_branch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
