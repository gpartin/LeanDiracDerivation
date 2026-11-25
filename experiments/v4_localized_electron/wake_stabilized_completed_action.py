"""Direct-prong transport plus the prior rest-null wake-control sector.

This is a diagnostic composition of already-derived native action terms.  It
tests whether the direct component handoff and the earlier Floquet-stable
receiving-shell response solve complementary problems.  Coefficients are
the earlier research values and are not promoted by this module.
"""

from __future__ import annotations

import os

import torch

import canonical_transfer_oscillator_gate as oscillator
import canonical_transfer_oscillator_implicit_update as wake
import completed_mixed_prong_action as transport


EVEN_TRANSFER_STRENGTH = float(os.environ.get("LFM_EVEN_TRANSFER", "2"))
DENSITY_BALANCE_STRENGTH = float(os.environ.get("LFM_DENSITY_BALANCE", "0.5"))
LOOP_CURVATURE_STRENGTH = float(os.environ.get("LFM_LOOP_CURVATURE", "1"))


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    result = transport.interval(left, right, dt)
    result = result + EVEN_TRANSFER_STRENGTH * oscillator.oscillator_interval(
        left, right, "transfer_even", dt)
    result = result - dt*DENSITY_BALANCE_STRENGTH*(
        wake.density_balance_energy(left)
        + wake.density_balance_energy(right))/2
    result = result - dt*LOOP_CURVATURE_STRENGTH*(
        wake.loop_curvature_energy(left)
        + wake.loop_curvature_energy(right))/2
    return result


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
