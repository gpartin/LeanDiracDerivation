"""Cheap rest/spectrum gate for a positive quartic completion of U_area.

The native quadratic C19 area bond makes the retained quarter-turn carrier a
six-direction phase saddle.  This file tests the smallest smooth quartic
face/edge completion found by the exact cube phase response problem.

For a bond let

  C = Re(W_i^dagger W_j),  s = (|W_i|^2+|W_j|^2)/2,
  P = |W_i|^2 |W_j|^2,
  V1 = s (s-C),             V2 = 2 (P-C^2).

Both V1 and V2 are nonnegative.  The exceptional completion is

  face:  V2/4,
  edge:  7 V1/4 - 3 V2/16.

The edge expression is also nonnegative: writing q=s/(|W_i||W_j|)>=1 and
x=C/(|W_i||W_j|) gives

  (7/4)q(q-x) - (3/8)(1-x^2) >= 0.

The rational coefficients are the minimum-L1 solution of the exact C19 cube
conditions: equal aligned phase stiffness on face and edge bonds and positive
curvature on every non-global phase direction of the retained twist.  This is
a research candidate only; movement and the PN barrier remain separate gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
import torch

import localized_stability as native


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def fields(vector: torch.Tensor, side: int):
    data = vector.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :]) / np.sqrt(native.B)
    return psi, chi


def potential(vector: torch.Tensor, side: int, lam: float, eta: float):
    psi, chi = fields(vector, side)
    p = torch.sum(torch.abs(psi)**2, dim=-1)
    c = torch.sum(torch.abs(chi)**2, dim=-1)
    value = torch.sum((c-native.R**2)**2/4 + c*p/2 + lam*p**2/4)
    area = torch.linalg.cross(chi, psi, dim=-1)
    area_norm = torch.sum(torch.abs(area)**2, dim=-1)
    for offset, weight in native.DIRECTIONS:
        shifts = tuple(-x for x in offset)
        neighbor = torch.roll(area, shifts, (0, 1, 2))
        neighbor_norm = torch.roll(area_norm, shifts, (0, 1, 2))

        # Existing quadratic U_area.  The factor 1/2 is the native directed
        # convention; the +/- direction list counts every undirected bond
        # twice.
        difference = neighbor-area
        value = value + weight*torch.sum(torch.abs(difference)**2)/(2*native.R**2)

        inner = torch.real(torch.sum(torch.conj(area)*neighbor, dim=-1))
        mean_norm = (area_norm+neighbor_norm)/2
        product_norm = area_norm*neighbor_norm
        v1 = mean_norm*(mean_norm-inner)
        v2 = 2*(product_norm-inner**2)
        if weight == 2:  # face bond
            exceptional = v2/4
        else:            # edge bond
            exceptional = 7*v1/4-3*v2/16
        value = value + eta*torch.sum(exceptional)/(2*native.R**6)
    return value


def rotation_matrix(side: int, omega_psi: float, omega_chi: float):
    size = 12*side**3
    result = np.empty((size, size))
    basis = np.zeros(size)
    psi_rates = np.asarray((omega_psi, omega_psi, omega_psi-omega_chi))
    chi_rates = np.asarray((omega_chi, omega_chi, 0.0))
    for column in range(size):
        basis[column] = 1
        dpsi, dchi = native.unpack(basis, side)
        result[:, column] = native.pack(1j*psi_rates*dpsi,
                                         1j*chi_rates*dchi)
        basis[column] = 0
    return result


def rest_and_spectrum(side: int, amplitude_sq: float, lam: float, eta: float):
    psi, chi = native.carrier_fields(side, amplitude_sq)
    q = torch.tensor(native.pack(psi, chi), requires_grad=True)
    energy = potential(q, side, lam, eta)
    gradient, = torch.autograd.grad(energy, q, create_graph=True)
    force = -gradient.detach().numpy()
    force_psi, force_chi = native.unpack(force, side)
    mask = native.norm2(psi) > 0
    active_chi = np.zeros_like(chi)
    active_chi[mask] = chi[mask]
    omega_psi_sq = float(-np.vdot(psi, force_psi).real/np.vdot(psi, psi).real)
    omega_chi_sq = float(-np.vdot(active_chi, force_chi).real
                         / np.vdot(active_chi, active_chi).real)
    if min(omega_psi_sq, omega_chi_sq) <= 0:
        return {
            "amplitude_sq": amplitude_sq,
            "lambda": lam,
            "eta": eta,
            "positive_frequencies": False,
            "omega_psi_sq": omega_psi_sq,
            "omega_chi_sq": omega_chi_sq,
        }
    omega_psi = float(np.sqrt(omega_psi_sq))
    omega_chi = float(np.sqrt(omega_chi_sq))
    gamma = rotation_matrix(side, omega_psi, omega_chi)
    rest_residual = force-gamma@gamma@q.detach().numpy()
    rest_relative = float(np.linalg.norm(rest_residual)
                          / max(np.linalg.norm(force), 1e-30))
    result = {
        "amplitude_sq": amplitude_sq,
        "lambda": lam,
        "eta": eta,
        "positive_frequencies": True,
        "omega_psi_sq": omega_psi_sq,
        "omega_chi_sq": omega_chi_sq,
        "rest_relative_residual": rest_relative,
    }
    if rest_relative > 1e-8:
        return result

    hessian = torch.autograd.functional.hessian(
        lambda x: potential(x, side, lam, eta), q,
        vectorize=True).detach().numpy()
    force_jacobian = -hessian
    size = hessian.shape[0]
    stiffness = gamma@gamma-force_jacobian
    generator = np.block((
        (np.zeros((size, size)), np.eye(size)),
        (-stiffness, -2*gamma),
    ))
    values = eig(generator, right=False, check_finite=False)
    result.update({
        "maximum_continuous_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "growth_count_above_1e_6": int(np.sum(values.real > 1e-6)),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--amplitude-sq", type=float,
                        default=native.R**2/(2+54/native.R**2))
    parser.add_argument("--lambda-self", nargs="+", type=float,
                        default=(1.0, 0.25))
    parser.add_argument("--eta", nargs="+", type=float,
                        default=(0.0, 2*native.R))
    parser.add_argument("--output", default="exceptional_bond_candidate.json")
    args = parser.parse_args()
    native.B = 1197.0
    rows = []
    for lam in args.lambda_self:
        for eta in args.eta:
            rows.append(rest_and_spectrum(
                args.side, args.amplitude_sq, lam, eta))
    result = {
        "status": "V4_EXCEPTIONAL_POSITIVE_BOND_CHEAP_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficient_provenance": (
            "exact minimum-L1 C19 phase-response solution; eta=2*chi0 is "
            "the no-new-constant geometric normalization tested first"
        ),
        "rows": rows,
        "claim_boundary": (
            "Exact-rest and full linear-spectrum gate only. A spectral pass "
            "would still require PN, movement, nonlinear, and Dirac tests."
        ),
    }
    destination = HERE/args.output
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
