"""C19 transport of the complete live Psi--chi relative frame.

The canonical area bond communicates chi cross Psi between neighboring sites.
For complex three-vectors, the complementary invariant chi^dagger Psi carries
the parallel complex response omitted by the wedge.  This module adds its
positive C19 bond square with the same 1/(2 chi0^2) normalization as the area
bond.  Together the two channels form the complete local relative frame.

The retained carrier has chi^dagger Psi=0 at every site, so this completion is
exactly rest-null.  It contains no prepared direction, corner label, target
state, projector, or Dirac object, and its variation is reciprocal in both
native fields.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def parallel_bond_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = local.unpack_tensor(raw, SIDE)
    # conj_physical avoids a lazy conjugate view whose second derivative has
    # no batched functorch rule on the supported PyTorch build.
    overlap = torch.sum(chi.conj_physical()*psi, dim=-1, keepdim=True)
    energy = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = overlap-common.neighbor(overlap, offset)
        energy = energy+weight*torch.sum(common.squared_norm(difference))
    return energy/(2*rank.R**2)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    completion = (parallel_bond_energy(left)+parallel_bond_energy(right))/2
    return incumbent.interval(left, right, dt)-dt*completion


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
