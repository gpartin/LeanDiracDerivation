"""Arbitrary-periodic-box form of the unchanged cube-incidence action.

The research terms were originally implemented with module-level ``SIDE=5``
constants.  Their formulas are already local periodic C19 sums.  This adapter
sets one declared side consistently across those implementations and composes
the exact same interval action.  It introduces no new term or coefficient.
"""

from __future__ import annotations

import torch

import area_bilinear_phase_action_gate as area_phase
import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_nonlinear_action as side5_action
import cube_incidence_symplectic_basis as cube_basis
import density_oriented_pair_connection as density_connection
import personal_touch_activated_bond_basis as activated


CUBE_COEFFICIENTS = side5_action.CUBE_COEFFICIENTS
DENSITY_COEFFICIENTS = side5_action.DENSITY_COEFFICIENTS


def configure_side(side: int) -> None:
    if side < 3:
        raise ValueError("periodic C19 box side must be at least three")
    # Each referenced formula obtains its shape only from this constant; its
    # C19 offsets, weights, fields and coefficients are unchanged.
    activated.cone.SIDE = side
    area_phase.SIDE = side
    density_connection.SIDE = side
    cube_basis.SIDE = side


def extra_potential(raw: torch.Tensor, side: int) -> torch.Tensor:
    configure_side(side)
    return (sum(activated.energies(raw))
            + area_phase.raw_potential(raw)/native_time.CHI0**14)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    side: int) -> torch.Tensor:
    configure_side(side)
    potential = native_time.DT*(
        extra_potential(left, side)+extra_potential(right, side))/2
    return (
        native_box.interval(left, right, side)-potential
        + native_time.B*native_time.CHI0*density_connection.interval(
            left, right, DENSITY_COEFFICIENTS)
        + cube_basis.interval(left, right, CUBE_COEFFICIENTS)
    )


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, side: int) -> torch.Tensor:
    value = (interval_action(previous, current, side)
             + interval_action(current, following, side))
    return torch.autograd.grad(value, current, create_graph=True)[0]
