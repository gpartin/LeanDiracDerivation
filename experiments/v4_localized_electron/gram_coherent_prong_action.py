"""Full live-metric coherence between prong and common material velocities.

The earlier coherence probe weighted disagreement only by the tiny numerical
regularizer.  Here each disagreement is measured by its complete native
Psi/chi gradient Gram matrix plus that regularizer.  The term is positive,
cubic symmetric, rest-null, and vanishes for exact common translation.  The
three prong contributions enter through their ordinary arithmetic mean.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import exact_translation_per_prong_gate as relative
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE
PRONG_AVERAGE = 1.0/3.0


def coherence_kinetic(psi: torch.Tensor, psi_dot: torch.Tensor,
                      chi: torch.Tensor, chi_dot: torch.Tensor) -> torch.Tensor:
    shared_velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")
    prong_velocity = relative.per_prong_velocity(
        psi, psi_dot, chi, chi_dot)
    psi_gradient = common.complex_gradient(psi)
    chi_gradient = common.complex_gradient(chi)
    gram = torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(psi_gradient), psi_gradient))
    gram += rank.B*torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(chi_gradient), chi_gradient))
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (relative.TRACE_REGULARIZER*trace/3
                   + relative.VACUUM_REGULARIZER)
    metric = gram+regularizer[..., None, None]*torch.eye(
        3, dtype=torch.float64)
    disagreement = prong_velocity-shared_velocity[..., None, :]
    squared = torch.einsum(
        "...ca,...cab,...cb->...c", disagreement, metric, disagreement)
    return PRONG_AVERAGE*torch.sum(squared)/2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    return (incumbent.interval(left, right, dt)
            + dt*coherence_kinetic(psi, psi_dot, chi, chi_dot))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
