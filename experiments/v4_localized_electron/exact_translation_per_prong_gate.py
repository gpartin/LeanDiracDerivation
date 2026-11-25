"""Per-prong refinement of the component-preserving material action.

Each matched pair (Psi_a, chi_a) derives its own local spatial velocity from
the same C19 gradients.  The velocity is eliminated algebraically, so no new
field or runtime track is introduced.  A common U(1) phase is removed for each
three-component field before the three prong solves; independent component
phases are not declared gauge.
"""

from __future__ import annotations

import torch

import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import exact_translation_multichannel_prong_gate as shared
import density_mold_connection_probe as local


SIDE = shared.SIDE
TRACE_REGULARIZER = rank.KAPPA
VACUUM_REGULARIZER = rank.KAPPA**3 * rank.R**2


def per_prong_velocity(psi: torch.Tensor, psi_dot: torch.Tensor,
                       chi: torch.Tensor, chi_dot: torch.Tensor) -> torch.Tensor:
    psi_gradient = shared.complex_gradient(psi)
    chi_gradient = shared.complex_gradient(chi)
    psi_reduced = shared.remove_phase_velocity(psi, psi_dot)
    chi_reduced = shared.remove_phase_velocity(chi, chi_dot)
    gram = torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(psi_gradient), psi_gradient))
    gram = gram + rank.B * torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(chi_gradient), chi_gradient))
    rhs = torch.real(torch.conj(psi_gradient) * psi_reduced[..., :, None])
    rhs = rhs + rank.B * torch.real(
        torch.conj(chi_gradient) * chi_reduced[..., :, None])
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    regularizer = TRACE_REGULARIZER * trace / 3 + VACUUM_REGULARIZER
    system = gram + regularizer[..., None, None] * identity
    return -torch.linalg.solve(system, rhs[..., None])[..., 0]


def advect(value: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        direction = torch.tensor(offset, dtype=torch.float64)
        velocity_mid = (
            velocity + shared.neighbor(velocity, offset)) / 2
        along = torch.einsum("...ca,a->...c", velocity_mid, direction)
        result = result + weight * along * (
            shared.neighbor(value, offset) - value)
    return result / shared.SECOND_MOMENT


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    velocity = per_prong_velocity(psi, psi_dot, chi, chi_dot)
    material_psi = psi_dot + advect(psi, velocity)
    material_chi = chi_dot + advect(chi, velocity)
    kinetic = torch.sum(shared.squared_norm(material_psi)) / 2
    kinetic = kinetic + rank.B * torch.sum(
        shared.squared_norm(material_chi)) / 2
    return dt * (kinetic - (four.orbit.potential(left)
                            + four.orbit.potential(right)) / 2)


def del_three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
