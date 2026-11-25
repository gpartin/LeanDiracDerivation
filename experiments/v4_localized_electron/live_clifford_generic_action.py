"""Arbitrary-box adapter for the live six-face research action.

Every configured module implements a local periodic formula whose historical
test harness stored the box side in a module constant.  This adapter changes
only that shared shape metadata, then evaluates the identical interval action.
It introduces no field, force, coefficient, or boundary rule.
"""

from __future__ import annotations

import torch

import canonical_transfer_oscillator_gate as oscillator
import canonical_transfer_oscillator_implicit_update as wake
import completed_mixed_prong_action as completed
import density_mold_connection_probe as local
import exact_translation_multichannel_prong_gate as common
import exact_translation_per_prong_gate as relative
import live_clifford_bond_action as live
import overlap_material_action as overlap_material
import personal_touch_unit_geometric_nonlinear as stable
import triangle_movement_action as triangle


def configure_side(side: int) -> None:
    if side < 5 or side % 2 == 0:
        raise ValueError("localized periodic box must be odd and at least five")
    # Explicit list: no recursive module mutation or hidden shadow config.
    local.SIDE = side
    live.SIDE = side
    completed.SIDE = side
    overlap_material.SIDE = side
    triangle.SIDE = side
    common.SIDE = side
    relative.SIDE = side
    common.precursor.SIDE = side
    common.inverse.SIDE = side
    common.four.orbit.SIDE = side
    stable.SIDE = side
    wake.SIDE = side
    wake.large.SIDE = side
    oscillator.large.SIDE = side


def interval(left: torch.Tensor, right: torch.Tensor,
             dt: float, side: int) -> torch.Tensor:
    configure_side(side)
    return live.interval(left, right, dt)


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float,
                 side: int) -> torch.Tensor:
    def summed(left, middle, right):
        return (interval(left, middle, dt, side)
                + interval(middle, right, dt, side))
    return torch.func.grad(summed, argnums=1)(previous, current, following)
