"""Movement-only cube-symplectic completion of the stable V4 action.

The incumbent action already preserves the localized resting carrier.  Its
first moving departure contains comparable components in the parallel
``chi^dagger Psi`` and transverse ``chi wedge Psi`` channels.  The reciprocal
cube-incidence one-form derived in ``cube_incidence_symplectic_basis`` is the
lowest existing native action term that mixes both responses while having
zero discrete Euler--Lagrange force on the retained rest orbit.

The internal coefficient vector is the exact integer null vector

  (-4, 0, +4, 0, -4, +1, 0, 0),

and the overall factor six is the smallest positive cube-geometric scale that
passed the complete native spectrum screen.  Neither value is fitted to the
Dirac comparator.  This is a research action, not a canonical promotion.
"""

from __future__ import annotations

import os

import torch

# Lock the incumbent configuration used by the accepted stable control before
# importing its module graph.  This is provenance, not a parameter search.
os.environ.setdefault("LFM_RELATIVE_PRONG_WEIGHT", "0")

import cube_incidence_symplectic_basis as cube
import cube_incidence_symplectic_completion_gate as completion
import wake_stabilized_completed_action as incumbent


CUBE_SCALE = float(os.environ.get("LFM_CUBE_SYMPLECTIC_SCALE", "1"))
CUBE_VARIANT = os.environ.get("LFM_CUBE_SYMPLECTIC_VARIANT", "overlap")
_COEFFICIENTS = {
    "gradient": completion.COEFFICIENTS,
    # Exact rest-force null directions of the complete native frame.
    "overlap": (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "common_relative": (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "complete": tuple(
        -0.5*float(gradient)+float(overlap)
        for gradient, overlap in zip(
            completion.COEFFICIENTS,
            (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            strict=True)),
}
CUBE_COEFFICIENTS = torch.tensor(
    _COEFFICIENTS[CUBE_VARIANT], dtype=torch.float64)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    return (incumbent.interval(left, right, dt)
            + CUBE_SCALE*cube.interval(left, right, CUBE_COEFFICIENTS))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt) + interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
