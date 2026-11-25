"""Stable common transport with a native relative-prong coherence metric.

Each matched Psi/chi prong may diagnose its own local material velocity, but
the action penalizes its disagreement with the single shared velocity instead
of granting three independent transport velocities.  The coefficient is the
ordinary average over the three native C3 prongs.  The term is positive,
reciprocal, cubic-symmetric, and rest-null.
"""

from __future__ import annotations

import torch

import completed_mixed_prong_action as completed
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
    gram = gram + rank.B*torch.real(torch.einsum(
        "...ca,...cb->...cab", torch.conj(chi_gradient), chi_gradient))
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (relative.TRACE_REGULARIZER*trace/3
                   + relative.VACUUM_REGULARIZER)
    disagreement = prong_velocity-shared_velocity[..., None, :]
    return PRONG_AVERAGE*torch.sum(
        regularizer*torch.sum(disagreement**2, dim=-1))/2


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
