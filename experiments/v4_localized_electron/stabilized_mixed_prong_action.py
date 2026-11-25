"""Mixed-prong transport kinetic plus the reduced stable V4 potential."""

from __future__ import annotations

import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import mixed_prong_action as transport
import personal_touch_unit_geometric_nonlinear as stable


COMMON_WEIGHT = transport.COMMON_WEIGHT
RELATIVE_WEIGHT = transport.RELATIVE_WEIGHT


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    tight_frame = dt * (
        stable.carrier_tight_frame_energy(left)
        + stable.carrier_tight_frame_energy(right)) / 2
    return transport.interval(left, right, dt) - tight_frame


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
