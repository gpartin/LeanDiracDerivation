"""Material connection of chi cross Psi without extra orientation inertia.

For w=(chi cross Psi)/chi0, add only the convective completion

  |w_dot + v.grad w|^2/2 - |w_dot|^2/2.

Thus the existing common velocity transports the relative orientation, while
no independent w kinetic mass, rest clock, direction, or target is added.
The full candidate kinetic metric must still pass positivity.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def oriented_area(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(chi, psi, dim=-1)/rank.R


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")
    w_left = oriented_area(left_psi, left_chi)
    w_right = oriented_area(right_psi, right_chi)
    w = (w_left+w_right)/2
    w_dot = (w_right-w_left)/dt
    advected = common.advect_same_components(w, velocity)
    connected = w_dot+advected
    completion = (
        torch.sum(common.squared_norm(connected))
        - torch.sum(common.squared_norm(w_dot)))/2
    return incumbent.interval(left, right, dt)+dt*completion


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
