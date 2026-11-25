"""Run the matched four-update orbit gate for the traceless tensor metric."""

from __future__ import annotations

import numpy as np
import torch

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four


def tensor_interval_action(left: torch.Tensor, right: torch.Tensor, dt: float,
                           coefficient: float) -> torch.Tensor:
    psi_left, chi_left = local.unpack_tensor(left, four.orbit.SIDE)
    psi_right, chi_right = local.unpack_tensor(right, four.orbit.SIDE)
    psi = (psi_left + psi_right) / 2
    chi = (chi_left + chi_right) / 2
    psi_dot = (psi_right - psi_left) / dt
    chi_dot = (chi_right - chi_left) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (chi_dot + local.ALPHA * response
                / (2 * denominator[..., None]))
    averaged = local.adjacency_average(velocity)
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(averaged)))
    return dt * (kinetic - (four.orbit.potential(left)
                            + four.orbit.potential(right)) / 2)


def main() -> None:
    original = four.orbit.interval_action
    four.orbit.interval_action = tensor_interval_action
    try:
        checkpoint = four.HERE / "edge_snap_tensor_four_update_orbit.npz"
        initial_middle = (np.load(checkpoint)["middle"]
                          if checkpoint.exists() else None)
        four.main(
            iterations=600,
            output_stem="edge_snap_tensor_four_update_orbit",
            action_variant="local_plus_traceless_density_mold",
            initial_middle=initial_middle)
    finally:
        four.orbit.interval_action = original


if __name__ == "__main__":
    main()
