"""Positive live-beta kinetic metric for the native Psi departure velocity.

The live relative edge grade is the cubical beta involution on the retained
carrier, derived without a fixed parity matrix.  The metric correction

    (eta/2) sum_i b_i |D_t Psi_i|^2,    eta = 1/chi0,

distinguishes the two beta grades only when Psi departs from its exact native
rest clock.  Since |b_i| <= 3 in a generic six-face neighborhood, the research
gate audits positivity explicitly; on the retained carrier |b_i|=1 and the
metric eigenvalues are 1 +/- 1/19.  Full variation includes the reciprocal chi
response through b_i.  No fixed beta, corner label, or Dirac object occurs.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import density_mold_connection_probe as local
import live_clifford_bond_action as incumbent
import live_relative_grade_action as grade


METRIC_SCALE = float(os.environ.get(
    "LFM_LIVE_GRADE_METRIC_SCALE", str(1/native_time.CHI0)))


def metric_energy(left: torch.Tensor, right: torch.Tensor,
                  dt: float) -> torch.Tensor:
    side = round((left.numel()/12)**(1/3))
    if side**3*12 != left.numel():
        raise ValueError("native state is not a cubic Psi/chi lattice")
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    centered_clock = 2*np.tan(theta/2)/dt
    departure = psi_dot-1j*centered_clock*psi
    departure_norm2 = torch.real(torch.sum(
        torch.conj(departure)*departure, dim=-1))
    return torch.sum(grade.live_grade(psi, chi)*departure_norm2)/2


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    return (incumbent.interval(left, right, dt)
            + dt*METRIC_SCALE*metric_energy(left, right, dt))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
