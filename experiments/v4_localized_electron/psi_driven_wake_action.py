"""Psi-derived common motion carrying both Psi and chi.

All three complex Psi prongs determine one phase-quotiented local translation
velocity.  The same velocity enters the material rates of both native fields,
so chi is transported by Psi's incoming pattern instead of dominating the
direction estimate through its large inertia B.  Reciprocity is retained:
the velocity is eliminated inside one real interval action, and variation of
that action still produces both Psi and chi equations.

The positive regularizer used in the velocity normal equation is included in
the interval kinetic, completing the Schur-complement action.  The existing
rest-null wake-control sector is retained unchanged for the matched test.
"""

from __future__ import annotations

import torch

import canonical_transfer_oscillator_gate as oscillator
import canonical_transfer_oscillator_implicit_update as wake
import density_mold_connection_probe as local
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


def psi_velocity_penalty(psi: torch.Tensor, psi_dot: torch.Tensor,
                         chi: torch.Tensor, chi_dot: torch.Tensor) -> torch.Tensor:
    del chi, chi_dot
    gram, _ = common.normal_blocks(psi, psi_dot, 1.0)
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (common.TRACE_REGULARIZER*trace/3
                   + common.VACUUM_REGULARIZER)
    # The chi arguments are unused by the source="psi" normal equation, but
    # the public function keeps one common interface for all native fields.
    zeros = torch.zeros_like(psi)
    velocity = common.best_matching_velocity(
        psi, psi_dot, zeros, zeros, "psi")
    return torch.sum(regularizer*torch.sum(velocity**2, dim=-1))/2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    if abs(dt-wake.small.DT) > 1e-15:
        raise ValueError("wake oscillator must be re-derived for this dt")
    # The common native action transports both fields; only the velocity vote
    # is sourced by Psi.
    result = common.interval(left, right, dt, "psi")
    left_psi, left_chi = local.unpack_tensor(left, common.SIDE)
    right_psi, right_chi = local.unpack_tensor(right, common.SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    result = result+dt*psi_velocity_penalty(psi, psi_dot, chi, chi_dot)
    result = result+incumbent.EVEN_TRANSFER_STRENGTH*oscillator.oscillator_interval(
        left, right, "transfer_even")
    result = result-dt*incumbent.DENSITY_BALANCE_STRENGTH*(
        wake.density_balance_energy(left)+wake.density_balance_energy(right))/2
    result = result-dt*incumbent.LOOP_CURVATURE_STRENGTH*(
        wake.loop_curvature_energy(left)+wake.loop_curvature_energy(right))/2
    return result


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
