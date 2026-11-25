"""Complete tangent test of the carrier/vacuum interface response.

The research action probe is

  U_interface = eta sum_(i,n in C19) w_n
      (rhoPsi_i-rhoPsi_(i+n))^2 |chi_i^dagger chi_(i+n)|^2 /(2 R^6).

It is real, positive, reciprocal, cubic, and uses only the native fields and
C19 bonds.  It has zero value and zero first derivative on the exact static
chi carrier because core-core bonds have equal Psi density and core-vacuum
chi vectors are orthogonal.  The handoff Fredholm audit shows that its second
derivative supplies the receiver-orientation channel missing from the
canonical action.  This script asks the separate question: does that same
term repair or damage the full rest spectrum?

No coefficient is admitted and no action is promoted by this scan.
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


def complex_norm_sq(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sum(value.real.square() + value.imag.square(), dim=dim)


def interface_energy(vector: torch.Tensor, side: int):
    data = vector.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :]) / np.sqrt(native.B)
    p = complex_norm_sq(psi)
    result = vector.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        shifts = tuple(-value for value in offset)
        neighbor_p = torch.roll(p, shifts, (0, 1, 2))
        neighbor_chi = torch.roll(chi, shifts, (0, 1, 2))
        overlap = torch.sum(chi.conj() * neighbor_chi, dim=-1)
        overlap_sq = overlap.real.square() + overlap.imag.square()
        result = result + weight * torch.sum(
            (p - neighbor_p).square() * overlap_sq
        ) / (2 * native.R**6)
    return result


def classify(side: int, eta: float, hessian: np.ndarray,
             psi: np.ndarray, chi: np.ndarray, force_jacobian: np.ndarray,
             gradient_norm: float):
    omega = native.R
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
        "eta": eta,
        "action_gradient_norm_at_orbit": gradient_norm * abs(eta),
        "maximum_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "constrained_hessian_index": index,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--etas", nargs="+", type=float,
                        default=(0.0, 1/1197, 1/19, 0.25, 1.0, 4.0,
                                 19.0, 38.0, 1197.0))
    parser.add_argument("--output", default="interface_stiffness_tangent.json")
    args = parser.parse_args()

    # canonical_constrained_index is historical research code and mutates the
    # shared module's B at import time.  This audit is explicitly V40/V4 and
    # must reset the canonical inertia before constructing mass-scaled fields.
    native.B = 1197.0

    amplitude_sq = native.R**2 / (2 + 54 / native.R**2)
    psi, chi = native.carrier_fields(args.side, amplitude_sq)
    fp, fc = native.force(psi, chi)
    dynamic_chi = np.zeros_like(chi)
    mask = native.norm2(psi) > 0
    dynamic_chi[mask] = chi[mask]
    psi_residual = np.linalg.norm(fp + native.R**2 * psi) / max(
        np.linalg.norm(fp), 1e-30)
    chi_residual = np.linalg.norm(fc) / max(
        np.linalg.norm(chi[mask]), 1e-30)

    configuration = native.pack(psi, chi)
    variable = torch.tensor(configuration, requires_grad=True)
    action_energy = interface_energy(variable, args.side)
    gradient, = torch.autograd.grad(action_energy, variable, create_graph=True)
    hessian = torch.autograd.functional.hessian(
        lambda x: interface_energy(x, args.side), variable,
        vectorize=True).detach().numpy()
    gradient = gradient.detach().numpy()
    force_jacobian = binding.modified_jacobian(psi, chi, 1.0, 1.0)
    rows = [classify(args.side, eta, hessian, psi, chi, force_jacobian,
                     float(np.linalg.norm(gradient))) for eta in args.etas]
    result = {
        "status": "V4_INTERFACE_STIFFNESS_COMPLETE_TANGENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": args.side,
        "B": native.B,
        "amplitude_sq": float(amplitude_sq),
        "psi_rest_residual": float(psi_residual),
        "chi_rest_residual": float(chi_residual),
        "interface_energy_at_orbit": float(action_energy.detach().numpy()),
        "interface_gradient_norm_at_orbit": float(np.linalg.norm(gradient)),
        "interface_hessian_symmetry_defect": float(np.linalg.norm(
            hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-30)),
        "rows": rows,
        "verdict": "TRANSPORT_RESPONSE_REST_SPECTRUM_CLASSIFIED",
        "claim_boundary": (
            "The interface term is a positive response probe with an "
            "unselected coefficient.  A favorable tangent result would still "
            "require nonlinear rest, translation, conservation, Dirac, and "
            "coefficient-provenance gates."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
