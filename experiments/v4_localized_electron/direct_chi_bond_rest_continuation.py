"""Continue the complete localized rest orbit under the direct-chi bond.

The optimization constraints fix only global phase, total Psi norm, and the
translation gauge used to select the branch.  They are not action terms or
runtime support.  Acceptance is evaluated with the unconstrained discrete
Euler--Lagrange residual after every continuation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import chi_transfer_softening_rest_spectrum as transfer_rest
import density_mold_connection_probe as local
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = transfer_rest.rest.SIDE
DT = transfer_rest.rest.DT
GEOMETRIC_FACTOR = 19.0


def pack_tensor(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def rotate(raw: torch.Tensor, theta_psi: torch.Tensor,
           theta_chi: torch.Tensor, sign: int) -> torch.Tensor:
    psi, chi = local.unpack_tensor(raw, SIDE)
    psi_phase = torch.complex(torch.cos(theta_psi),
                              sign * torch.sin(theta_psi))
    chi_phase = torch.complex(torch.cos(theta_chi),
                              sign * torch.sin(theta_chi))
    return pack_tensor(psi * psi_phase, chi * chi_phase)


def direct_chi_bond(raw: torch.Tensor) -> torch.Tensor:
    _, chi = local.unpack_tensor(raw, SIDE)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = chi - torch.roll(
            chi, shifts=offset, dims=(0, 1, 2))
        bond = bond + weight * torch.sum(local.squared_norm(difference))
    return (rank.B / 6) * bond / (2 * rank.C19_WEIGHT_SUM)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    continuation: float) -> torch.Tensor:
    baseline = transfer_rest.interval_action(
        left, right, GEOMETRIC_FACTOR)
    return baseline - DT * continuation * (
        direct_chi_bond(left) + direct_chi_bond(right)) / 2


def del_residual(current: torch.Tensor, theta_psi: torch.Tensor,
                 theta_chi: torch.Tensor, continuation: float) -> torch.Tensor:
    previous = rotate(current, theta_psi, theta_chi, -1)
    following = rotate(current, theta_psi, theta_chi, 1)
    # Differentiate with respect to an independent copy so the returned DEL
    # is the partial derivative at the middle layer, while retaining the
    # outer graph through the relative-periodic previous/following values.
    independent = current.clone()
    if not independent.requires_grad:
        independent.requires_grad_(True)
    action = (interval_action(previous, independent, continuation)
              + interval_action(independent, following, continuation))
    return torch.autograd.grad(
        action, independent, create_graph=True)[0]


def metrics(current: torch.Tensor, reference: torch.Tensor,
            theta_psi: torch.Tensor, theta_chi: torch.Tensor,
            continuation: float) -> dict:
    psi, chi = local.unpack_tensor(current, SIDE)
    psi_reference, chi_reference = local.unpack_tensor(reference, SIDE)
    residual = del_residual(
        current, theta_psi, theta_chi, continuation).detach()
    scale = rank.B * rank.R / DT
    mask = torch.sum(local.squared_norm(psi_reference), dim=-1) > 1e-14
    total = torch.sum(local.squared_norm(psi))
    exterior = torch.sum(local.squared_norm(psi[~mask]))
    overlap = abs(torch.sum(torch.conj(psi_reference[mask]) * psi[mask]))**2
    overlap = overlap / (
        torch.sum(local.squared_norm(psi_reference[mask]))
        * torch.sum(local.squared_norm(psi[mask])))
    return {
        "continuation": continuation,
        "normalized_del_rms": float(
            torch.linalg.vector_norm(residual)
            / np.sqrt(residual.numel()) / scale),
        "raw_del_rms": float(
            torch.linalg.vector_norm(residual) / np.sqrt(residual.numel())),
        "theta_psi": float(theta_psi.detach()),
        "theta_chi": float(theta_chi.detach()),
        "psi_exterior_fraction": float((exterior / total).detach()),
        "psi_carrier_overlap_sq": float(overlap.detach()),
        "relative_state_deformation": float((
            torch.linalg.vector_norm(current - reference)
            / torch.linalg.vector_norm(reference)).detach()),
        "chi_relative_deformation": float((
            torch.linalg.vector_norm(chi - chi_reference)
            / torch.linalg.vector_norm(chi_reference)).detach()),
    }


def main(steps: int = 10, iterations: int = 120) -> None:
    rank.lock_inertia()
    layers, theta = transfer_rest.rest.orbit_layers()
    reference = layers[1].detach()
    current = reference.clone().requires_grad_(True)
    theta_psi = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    theta_chi = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    psi_reference, chi_reference = local.unpack_tensor(reference, SIDE)
    reference_norm = torch.sum(local.squared_norm(psi_reference)).detach()
    coordinates = torch.arange(SIDE, dtype=torch.float64)
    coordinate_grids = torch.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij")
    reference_rho = torch.sum(local.squared_norm(psi_reference), dim=-1)
    reference_centers = torch.stack([
        torch.sum(reference_rho * grid) / reference_norm
        for grid in coordinate_grids])
    rows = []

    for index in range(steps + 1):
        continuation = index / steps
        if index:
            optimizer = torch.optim.LBFGS(
                [current, theta_psi, theta_chi], lr=0.5,
                max_iter=iterations, max_eval=iterations * 2,
                tolerance_grad=1e-12, tolerance_change=1e-15,
                history_size=40, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                residual = del_residual(
                    current, theta_psi, theta_chi, continuation)
                # The physical DEL normalization contains 1/DT and makes an
                # already unacceptable 1e-6 residual numerically invisible
                # to an optimizer.  Removing that harmless common factor
                # conditions the solve without changing its zeros.
                scale = rank.R
                psi, chi = local.unpack_tensor(current, SIDE)
                rho = torch.sum(local.squared_norm(psi), dim=-1)
                norm_defect = (torch.sum(rho) / reference_norm - 1)
                centers = torch.stack([
                    torch.sum(rho * grid) / torch.sum(rho)
                    for grid in coordinate_grids])
                center_defect = centers - reference_centers
                psi_gauge = torch.imag(torch.sum(
                    torch.conj(psi_reference) * psi)) / reference_norm
                chi_scale = torch.sum(local.squared_norm(chi_reference))
                chi_gauge = torch.imag(torch.sum(
                    torch.conj(chi_reference) * chi)) / chi_scale
                objective = torch.mean((residual / scale)**2)
                objective = objective + 10 * (
                    norm_defect**2 + torch.sum(center_defect**2)
                    + psi_gauge**2 + chi_gauge**2)
                objective.backward()
                return objective

            optimizer.step(closure)
        rows.append(metrics(
            current, reference, theta_psi, theta_chi, continuation))

    final = rows[-1]
    identity_retained = (
        final["psi_carrier_overlap_sq"] > 0.95
        and final["psi_exterior_fraction"] < 0.1)
    exact = final["normalized_del_rms"] < 1e-8
    result = {
        "status": "V4_DIRECT_CHI_BOND_REST_CONTINUATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "continuation*(B/6)*sum_C19 w|chi_(i+n)-chi_i|^2/(2*24)"),
        "side": SIDE,
        "dt": DT,
        "rows": rows,
        "exact_rest_orbit_found": exact,
        "carrier_identity_retained": identity_retained,
        "verdict": (
            "ADVANCE_TO_FULL_REST_SPECTRUM" if exact and identity_retained
            else "REST_CONTINUATION_NOT_CLOSED"),
        "claim_boundary": (
            "The norm, phase, and center conditions select a continuation "
            "branch only. They are not action terms or runtime support. A "
            "closed rest solve would still require a full spectrum gate."),
    }
    (HERE / "direct_chi_bond_rest_continuation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        HERE / "direct_chi_bond_rest_continuation.npz",
        current=current.detach().numpy(),
        theta_psi=float(theta_psi.detach()),
        theta_chi=float(theta_chi.detach()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    main(args.steps, args.iterations)
