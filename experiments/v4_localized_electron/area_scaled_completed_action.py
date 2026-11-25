"""Diagnostic C19 area-stiffness family around the completed action.

Only the coefficient of the already-native positive area-bond energy is
varied.  This is an obstruction/response scan, not an admitted coefficient.
"""

from __future__ import annotations

import os

import torch

import completed_mixed_prong_action as incumbent
import density_mold_connection_probe as local
import edge_snap_action_rank as rank


AREA_SCALE = float(os.environ.get("LFM_AREA_SCALE", "1"))
SIDE = incumbent.SIDE


def area_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = local.unpack_tensor(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        result = result + weight * torch.sum(local.squared_norm(difference))
    return result/(2*rank.R**2)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    delta = dt*(AREA_SCALE-1)*(area_energy(left)+area_energy(right))/2
    return incumbent.interval(left, right, dt)-delta


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
