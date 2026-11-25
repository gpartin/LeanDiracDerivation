"""One material velocity jointly inferred from Psi, chi, and chi cross Psi.

Unlike ``wedge_material_action``, the oriented area participates in the same
normal equation that determines transport.  Its material kinetic replaces the
standalone add-on construction, avoiding a velocity mismatch and double use of
the old metric.  The normalized area has the same native field dimension as
Psi and therefore carries unit geometric weight.
"""

from __future__ import annotations

import torch

import canonical_transfer_oscillator_gate as oscillator
import canonical_transfer_oscillator_implicit_update as wake
import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import exact_translation_multichannel_prong_gate as common
import personal_touch_unit_geometric_nonlinear as stable


SIDE = common.SIDE
WEDGE_SPATIAL_EIGENVALUE = 54.0
WEDGE_LOCAL_FREQUENCY_SQUARED = rank.R**2-WEDGE_SPATIAL_EIGENVALUE


def oriented_area(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(chi, psi, dim=-1)/rank.R


def oriented_area_bond_energy(w: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = w-torch.roll(w, shifts=offset, dims=(0, 1, 2))
        result += weight*torch.sum(common.squared_norm(difference))
    return result/2


def velocity_and_regularizer(psi, psi_dot, chi, chi_dot, w, w_dot):
    shape = psi.shape[:-1]
    gram = torch.zeros(shape+(3, 3), dtype=torch.float64)
    rhs = torch.zeros(shape+(3,), dtype=torch.float64)
    for value, derivative, weight in (
            (psi, psi_dot, 1.0), (chi, chi_dot, rank.B), (w, w_dot, 1.0)):
        addition, vector = common.normal_blocks(value, derivative, weight)
        gram, rhs = gram+addition, rhs+vector
    trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1)
    regularizer = (common.TRACE_REGULARIZER*trace/3
                   + common.VACUUM_REGULARIZER)
    identity = torch.eye(3, dtype=torch.float64)
    velocity = -torch.linalg.solve(
        gram+regularizer[..., None, None]*identity,
        rhs[..., None])[..., 0]
    return velocity, regularizer


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    if abs(dt-wake.small.DT) > 1e-15:
        raise ValueError("wake oscillator must be re-derived for this dt")
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    w_left = oriented_area(left_psi, left_chi)
    w_right = oriented_area(right_psi, right_chi)
    w = (w_left+w_right)/2
    w_dot = (w_right-w_left)/dt
    velocity, regularizer = velocity_and_regularizer(
        psi, psi_dot, chi, chi_dot, w, w_dot)

    material_psi = psi_dot+common.advect_same_components(psi, velocity)
    material_chi = chi_dot+common.advect_same_components(chi, velocity)
    material_w = w_dot+common.advect_same_components(w, velocity)
    kinetic = torch.sum(common.squared_norm(material_psi))/2
    kinetic += rank.B*torch.sum(common.squared_norm(material_chi))/2
    kinetic += torch.sum(common.squared_norm(material_w))/2
    kinetic += torch.sum(regularizer*torch.sum(velocity**2, dim=-1))/2

    potential = (four.orbit.potential(left)+four.orbit.potential(right))/2
    potential += (stable.carrier_tight_frame_energy(left)
                  + stable.carrier_tight_frame_energy(right))/2
    potential += WEDGE_LOCAL_FREQUENCY_SQUARED*torch.sum(
        common.squared_norm(w_left)+common.squared_norm(w_right))/4
    potential += (oriented_area_bond_energy(w_left)
                  + oriented_area_bond_energy(w_right))/2
    result = dt*(kinetic-potential)
    result += 2.0*oscillator.oscillator_interval(
        left, right, "transfer_even")
    result -= dt*0.5*(wake.density_balance_energy(left)
                     + wake.density_balance_energy(right))/2
    result -= dt*(wake.loop_curvature_energy(left)
                  + wake.loop_curvature_energy(right))/2
    return result


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
