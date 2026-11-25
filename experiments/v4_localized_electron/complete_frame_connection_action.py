"""Signed material connection for the complete live Psi--chi frame.

For both relative-frame coordinates

    w = (chi cross Psi)/chi0,   s = chi^dagger Psi/chi0,

add only the convective completion

    (|q_dot + v.grad q|^2 - |q_dot|^2)/2.

The cross term is odd in the native common velocity and therefore distinguishes
sender from receiver.  Subtracting |q_dot|^2 avoids adding a second intrinsic
frame inertia.  The term is rest-null, cubic, local over C19, reciprocal, and
contains no prepared direction or Dirac object.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def frame_coordinates(psi: torch.Tensor,
                      chi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.linalg.cross(chi, psi, dim=-1)/rank.R
    s = torch.sum(chi.conj_physical()*psi, dim=-1, keepdim=True)/rank.R
    return w, s


def connection_completion(left_q: torch.Tensor, right_q: torch.Tensor,
                          velocity: torch.Tensor, dt: float) -> torch.Tensor:
    q = (left_q+right_q)/2
    q_dot = (right_q-left_q)/dt
    advected = common.advect_same_components(q, velocity)
    material = q_dot+advected
    return (torch.sum(common.squared_norm(material))
            - torch.sum(common.squared_norm(q_dot)))/2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")
    left_w, left_s = frame_coordinates(left_psi, left_chi)
    right_w, right_s = frame_coordinates(right_psi, right_chi)
    completion = connection_completion(left_w, right_w, velocity, dt)
    completion = completion+connection_completion(
        left_s, right_s, velocity, dt)
    return incumbent.interval(left, right, dt)+dt*completion


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
