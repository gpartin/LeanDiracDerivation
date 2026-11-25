"""Combined reciprocal movement completion of the stable V4 action.

This is one action containing the rest-null curl--overlap positive square and
the exact complete cube-symplectic one-form.  The latter uses the C19
face/edge half-weight and the unit complete-frame overlap normalization.
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("LFM_CUBE_SYMPLECTIC_VARIANT", "complete")
os.environ.setdefault("LFM_CUBE_SYMPLECTIC_SCALE", "1")

import cube_incidence_symplectic_basis as cube
import cube_symplectic_movement_action as symplectic
import curl_overlap_movement_action as coherent

COHERENT_SCALE = float(os.environ.get("LFM_COHERENT_MOVEMENT_SCALE", "1"))


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    base = coherent.incumbent.interval(left, right, dt)
    return (base+COHERENT_SCALE*(coherent.interval(left, right, dt)-base)
            + symplectic.CUBE_SCALE*cube.interval(
                left, right, symplectic.CUBE_COEFFICIENTS))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
