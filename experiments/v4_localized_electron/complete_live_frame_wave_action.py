"""Complete material wave for the native relative Psi--chi frame.

The transverse coordinate w=(chi cross Psi)/chi0 and the parallel coordinate
s=chi^dagger Psi/chi0 together retain the full relative three-component
information.  The canonical action already contains the C19 spatial bond for
w.  ``wedge_material_action`` supplies its material kinetic and native rest
clock.  Here s receives its matching material kinetic and C19 spatial bond.
Because s=0 on the exact rest carrier, the added sector is rest-null.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wedge_material_action as transverse


SIDE = common.SIDE


def overlap(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.sum(chi.conj_physical()*psi, dim=-1, keepdim=True)/rank.R


def overlap_bond(s: torch.Tensor) -> torch.Tensor:
    energy = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = s-common.neighbor(s, offset)
        energy = energy+weight*torch.sum(common.squared_norm(difference))
    return energy/2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")

    s_left = overlap(left_psi, left_chi)
    s_right = overlap(right_psi, right_chi)
    s_mid = (s_left+s_right)/2
    s_dot = (s_right-s_left)/dt
    material_s = s_dot+common.advect_same_components(s_mid, velocity)
    kinetic = torch.sum(common.squared_norm(material_s))/2
    spatial = (overlap_bond(s_left)+overlap_bond(s_right))/2
    return transverse.interval(left, right, dt)+dt*(kinetic-spatial)


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
