"""Common transport completed by the induced Psi--chi oriented-area clock.

The native composite w=(chi cross Psi)/chi0 is not a new field.  Its material
rate uses the same common velocity already derived from all six complex
prongs.  A discrete harmonic potential at the native chi0 clock makes the
resting rotating w orbit an exact solution of this added reciprocal action.
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
    w_mid = (w_left+w_right)/2
    w_dot = (w_right-w_left)/dt
    material_w = w_dot+common.advect_same_components(w_mid, velocity)
    kinetic = torch.sum(common.squared_norm(material_w))/2
    potential = rank.R**2*torch.sum(
        common.squared_norm(w_left)+common.squared_norm(w_right))/4
    return incumbent.interval(left, right, dt)+dt*(kinetic-potential)


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
