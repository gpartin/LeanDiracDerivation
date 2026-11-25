"""Variational completion of the mixed direct-component material action.

The local best-matching velocities are solved with a positive Tikhonov
metric.  Earlier probes used that metric in the algebraic solve but omitted
its matching ``lambda |v|^2 / 2`` contribution from the interval action.
This module restores exactly that contribution.  It introduces no new field,
direction, carrier label, Dirac object, or fitted coefficient.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import exact_translation_per_prong_gate as relative
import stabilized_mixed_prong_action as incumbent


COMMON_WEIGHT = incumbent.COMMON_WEIGHT
RELATIVE_WEIGHT = incumbent.RELATIVE_WEIGHT
SIDE = common.SIDE


def shared_velocity_penalty(psi: torch.Tensor, psi_dot: torch.Tensor,
                            chi: torch.Tensor,
                            chi_dot: torch.Tensor) -> torch.Tensor:
    gram_psi, _ = common.normal_blocks(psi, psi_dot, 1.0)
    gram_chi, _ = common.normal_blocks(chi, chi_dot, rank.B)
    gram = gram_psi + gram_chi
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (common.TRACE_REGULARIZER * trace / 3
                   + common.VACUUM_REGULARIZER)
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")
    return torch.sum(regularizer * torch.sum(velocity**2, dim=-1)) / 2


def relative_velocity_penalty(psi: torch.Tensor, psi_dot: torch.Tensor,
                              chi: torch.Tensor,
                              chi_dot: torch.Tensor) -> torch.Tensor:
    psi_gradient = common.complex_gradient(psi)
    chi_gradient = common.complex_gradient(chi)
    gram = torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(psi_gradient), psi_gradient))
    gram = gram + rank.B * torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(chi_gradient), chi_gradient))
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (relative.TRACE_REGULARIZER * trace / 3
                   + relative.VACUUM_REGULARIZER)
    velocity = relative.per_prong_velocity(
        psi, psi_dot, chi, chi_dot)
    return torch.sum(regularizer * torch.sum(velocity**2, dim=-1)) / 2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    penalty = COMMON_WEIGHT * shared_velocity_penalty(
        psi, psi_dot, chi, chi_dot)
    penalty = penalty + RELATIVE_WEIGHT * relative_velocity_penalty(
        psi, psi_dot, chi, chi_dot)
    return incumbent.interval(left, right, dt) + dt * penalty


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
