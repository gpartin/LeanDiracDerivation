"""Four-update control for nonlinear χ density-transfer softening.

This retains the historical rigid endpoint only as the squared-response
control shared by the conditional edge channels.  A closed result would still
need the signed conditional readout gate; a failure is an action diagnostic.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import chi_bond_frame_transport_probe as bond_frame


GEOMETRIC_FACTOR = 6.0
FILTER_FACTOR = 0.0
COLLECTIVE_FACTOR = 0.0
RADIAL_FACTOR = 0.0
NORMAL_FACTOR = -1.0
BOND_FRAME_FACTOR = 0.0
TRANSFER_NEIGHBOR_NORMAL = 0.0
TRANSFER_NEIGHBOR_RADIAL = 0.0


def c19_dirichlet(value: torch.Tensor) -> torch.Tensor:
    return value * rank.C19_WEIGHT_SUM - sum(
        weight * torch.roll(value, shifts=offset, dims=(0, 1, 2))
        for offset, weight in rank.native.DIRECTIONS)


def interval_action(left: torch.Tensor, right: torch.Tensor, dt: float,
                    coefficient: float) -> torch.Tensor:
    del coefficient
    psi_left, chi_left = local.unpack_tensor(left, four.orbit.SIDE)
    psi_right, chi_right = local.unpack_tensor(right, four.orbit.SIDE)
    psi = (psi_left + psi_right) / 2
    chi = (chi_left + chi_right) / 2
    psi_dot = (psi_right - psi_left) / dt
    chi_dot = (chi_right - chi_left) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    chi_denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * chi_denominator[..., None])
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    if (TRANSFER_NEIGHBOR_NORMAL != 0
            or TRANSFER_NEIGHBOR_RADIAL != 0):
        chi_curvature = chi - local.adjacency_average(chi)
        curvature_overlap = torch.sum(
            torch.conj(chi) * chi_curvature, dim=-1)
        curvature_radial = (chi * curvature_overlap[..., None]
                            / chi_denominator[..., None])
        curvature_normal = chi_curvature - curvature_radial
        material = (material + rank.R * transfer[..., None]
                    * (TRANSFER_NEIGHBOR_NORMAL * curvature_normal
                       + TRANSFER_NEIGHBOR_RADIAL * curvature_radial))
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / chi_denominator[..., None]
    transverse = material - longitudinal
    scale = torch.rsqrt(1 + GEOMETRIC_FACTOR * transfer**2)
    radial_scale = torch.rsqrt(1 + RADIAL_FACTOR * transfer**2)
    if NORMAL_FACTOR >= 0:
        regulator = rank.KAPPA * rank.R**2
        chi_psi_overlap = torch.sum(torch.conj(chi) * psi, dim=-1)
        psi_orthogonal = (psi - chi * chi_psi_overlap[..., None]
                          / chi_denominator[..., None])
        rho_plane = torch.sum(local.squared_norm(psi_orthogonal), dim=-1)
        plane_denominator = rho_plane + regulator
        plane_overlap = torch.sum(
            torch.conj(psi_orthogonal) * transverse, dim=-1)
        plane = (psi_orthogonal * plane_overlap[..., None]
                 / plane_denominator[..., None])
        normal = transverse - plane
        normal_scale = torch.rsqrt(1 + NORMAL_FACTOR * transfer**2)
        framed = (radial_scale[..., None] * longitudinal
                  + scale[..., None] * plane
                  + normal_scale[..., None] * normal)
    else:
        framed = (radial_scale[..., None] * longitudinal
                  + scale[..., None] * transverse)
    averaged = local.adjacency_average(framed)
    if BOND_FRAME_FACTOR > 0:
        transported = bond_frame.transported_average(psi, chi, framed)
        bond_activation = (BOND_FRAME_FACTOR * transfer**2
                           / (1 + BOND_FRAME_FACTOR * transfer**2))
        averaged = ((1 - bond_activation)[..., None] * averaged
                    + bond_activation[..., None] * transported)
    collective_scale = torch.rsqrt(1 + COLLECTIVE_FACTOR * transfer**2)
    averaged = collective_scale[..., None] * averaged
    dirichlet = c19_dirichlet(framed) / rank.C19_WEIGHT_SUM
    activation = (FILTER_FACTOR * transfer**2
                  / (1 + FILTER_FACTOR * transfer**2))
    averaged = averaged + activation[..., None] * dirichlet
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(averaged)))
    return dt * (kinetic - (four.orbit.potential(left)
                            + four.orbit.potential(right)) / 2)


def main(iterations: int = 600, charge_weight: float = 1e-4,
         output_stem: str = "edge_snap_transfer_four_update_orbit",
         checkpoint_name: str = "edge_snap_tensor_four_update_orbit.npz",
         geometric_factor: float = 6.0,
         filter_factor: float = 0.0,
         collective_factor: float = 0.0,
         radial_factor: float = 0.0,
         normal_factor: float = -1.0,
         bond_frame_factor: float = 0.0,
         transfer_neighbor_normal: float = 0.0,
         transfer_neighbor_radial: float = 0.0) -> None:
    global GEOMETRIC_FACTOR, FILTER_FACTOR, COLLECTIVE_FACTOR, RADIAL_FACTOR
    global NORMAL_FACTOR
    global BOND_FRAME_FACTOR, TRANSFER_NEIGHBOR_NORMAL
    global TRANSFER_NEIGHBOR_RADIAL
    GEOMETRIC_FACTOR = geometric_factor
    FILTER_FACTOR = filter_factor
    COLLECTIVE_FACTOR = collective_factor
    RADIAL_FACTOR = radial_factor
    NORMAL_FACTOR = normal_factor
    BOND_FRAME_FACTOR = bond_frame_factor
    TRANSFER_NEIGHBOR_NORMAL = transfer_neighbor_normal
    TRANSFER_NEIGHBOR_RADIAL = transfer_neighbor_radial
    original = four.orbit.interval_action
    original_charge_weight = four.CHARGE_WEIGHT
    four.orbit.interval_action = interval_action
    four.CHARGE_WEIGHT = charge_weight
    try:
        checkpoint = four.HERE / checkpoint_name
        initial_middle = (np.load(checkpoint)["middle"]
                          if checkpoint.exists() else None)
        four.main(
            iterations=iterations,
            output_stem=output_stem,
            action_variant=(
                f"chi_transfer_softening_factor_{geometric_factor:g}_"
                f"face_filter_{filter_factor:g}_"
                f"collective_softening_{collective_factor:g}_"
                f"radial_softening_{radial_factor:g}_"
                f"normal_softening_{normal_factor:g}_"
                f"bond_frame_{bond_frame_factor:g}_"
                f"transfer_neighbor_normal_{transfer_neighbor_normal:g}_"
                f"transfer_neighbor_radial_{transfer_neighbor_radial:g}"),
            initial_middle=initial_middle)
    finally:
        four.orbit.interval_action = original
        four.CHARGE_WEIGHT = original_charge_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--charge-weight", type=float, default=1e-4)
    parser.add_argument(
        "--output-stem", default="edge_snap_transfer_four_update_orbit")
    parser.add_argument(
        "--checkpoint", default="edge_snap_tensor_four_update_orbit.npz")
    parser.add_argument("--geometric-factor", type=float, default=6.0)
    parser.add_argument("--filter-factor", type=float, default=0.0)
    parser.add_argument("--collective-factor", type=float, default=0.0)
    parser.add_argument("--radial-factor", type=float, default=0.0)
    parser.add_argument("--normal-factor", type=float, default=-1.0)
    parser.add_argument("--bond-frame-factor", type=float, default=0.0)
    parser.add_argument("--transfer-neighbor-normal", type=float, default=0.0)
    parser.add_argument("--transfer-neighbor-radial", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.iterations, args.charge_weight, args.output_stem,
         args.checkpoint, args.geometric_factor, args.filter_factor,
         args.collective_factor, args.radial_factor, args.normal_factor,
         args.bond_frame_factor, args.transfer_neighbor_normal,
         args.transfer_neighbor_radial)
