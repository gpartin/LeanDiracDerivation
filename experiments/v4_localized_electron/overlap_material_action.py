"""Material transport of the live parallel Psi--chi response.

The oriented area chi cross Psi records the transverse part of Psi relative
to chi but omits the parallel complex response.  The native scalar

    s = chi^dagger Psi / chi0

supplies that missing information.  On the retained rest carrier s and its
material rate vanish, so the positive kinetic below is exactly rest-null.
It is also zero under exact common translation.  Together s and chi cross
Psi form the complete live relative frame; no prepared ray or Dirac object is
used.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def parallel_overlap(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.conj(chi)*psi, dim=-1, keepdim=True)/rank.R


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")

    s_left = parallel_overlap(left_psi, left_chi)
    s_right = parallel_overlap(right_psi, right_chi)
    s_mid = (s_left+s_right)/2
    s_dot = (s_right-s_left)/dt
    material_s = s_dot+common.advect_same_components(s_mid, velocity)
    kinetic = torch.sum(common.squared_norm(material_s))/2
    return incumbent.interval(left, right, dt)+dt*kinetic


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
