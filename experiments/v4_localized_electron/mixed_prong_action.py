"""Geometry-weighted common/relative component material action.

One common three-vector communication channel and three component-pair
channels are counted equally, giving weights 1/4 and 3/4.  Both terms are
positive reciprocal interval actions with the same position potential; their
convex sum therefore keeps one native kinetic and potential normalization.
"""

from __future__ import annotations

import os

import torch

import exact_translation_multichannel_prong_gate as common
import exact_translation_per_prong_gate as relative


RELATIVE_WEIGHT = float(os.environ.get("LFM_RELATIVE_PRONG_WEIGHT", "0.75"))
COMMON_WEIGHT = 1.0-RELATIVE_WEIGHT


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    return (COMMON_WEIGHT * common.interval(left, right, dt, "both")
            + RELATIVE_WEIGHT * relative.interval(left, right, dt))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
