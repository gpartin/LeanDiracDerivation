"""Psi-led transport with density-activated chi frame completion.

Psi determines the common translation wherever its gradient has rank.  The
live chi gradient fills missing directions only in proportion to the local
Psi density, rho_Psi/chi0^2.  Thus vacuum chi cannot steer a carrier and the
large chi kinetic normalization B does not dominate the direction estimate.
The resulting velocity is substituted into one positive material kinetic for
both native fields; varying that real action remains reciprocal.
"""

from __future__ import annotations

import torch

import canonical_transfer_oscillator_gate as oscillator
import canonical_transfer_oscillator_implicit_update as wake
import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def density_completed_velocity(psi: torch.Tensor, psi_dot: torch.Tensor,
                               chi: torch.Tensor,
                               chi_dot: torch.Tensor) -> torch.Tensor:
    gram_psi, rhs_psi = common.normal_blocks(psi, psi_dot, 1.0)
    gram_chi, rhs_chi = common.normal_blocks(chi, chi_dot, 1.0)
    rho_psi = common.squared_norm(psi)
    activation = rho_psi/rank.R**2
    gram = gram_psi+activation[..., None, None]*gram_chi
    rhs = rhs_psi+activation[..., None]*rhs_chi
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (common.TRACE_REGULARIZER*trace/3
                   + common.VACUUM_REGULARIZER)
    system = gram+regularizer[..., None, None]*torch.eye(
        3, dtype=torch.float64)
    return -torch.linalg.solve(system, rhs[..., None])[..., 0]


def transport_interval(left: torch.Tensor, right: torch.Tensor,
                       dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = density_completed_velocity(psi, psi_dot, chi, chi_dot)
    material_psi = psi_dot+common.advect_same_components(psi, velocity)
    material_chi = chi_dot+common.advect_same_components(chi, velocity)
    kinetic = torch.sum(common.squared_norm(material_psi))/2
    kinetic = kinetic+rank.B*torch.sum(common.squared_norm(material_chi))/2
    potential = (four.orbit.potential(left)+four.orbit.potential(right))/2
    return dt*(kinetic-potential)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    if abs(dt-wake.small.DT) > 1e-15:
        raise ValueError("wake oscillator must be re-derived for this dt")
    result = transport_interval(left, right, dt)
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
