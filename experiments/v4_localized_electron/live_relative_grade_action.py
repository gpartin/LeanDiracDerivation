"""Rest-covariant action term from the live relative Psi--chi edge grade.

For every face neighbor, compare the unit overlap phase carried by Psi with
the unit overlap phase carried by chi.  The cubic sum

    b_i = sum_(face neighbors j) Re[i u^Psi_ij conj(u^chi_ij)]

is a native real local scalar.  On the retained carrier it equals the cubical
mass grading, but no corner labels, parity field, projector, or Dirac matrix
occurs in this action.  Coupling b_i to the Psi phase velocity relative to the
exact native rest clock makes the term vanish on the retained rest orbit while
responding to the opposite-frequency component needed during transport.

This is an unadmitted research action candidate.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import density_mold_connection_probe as local
import live_clifford_bond_action as incumbent


GRADE_SCALE = float(os.environ.get("LFM_LIVE_GRADE_SCALE", "1"))


def live_grade(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(psi.shape[:-1], dtype=torch.float64)
    for axis in range(3):
        for direction in (-1, 1):
            shifts = [0, 0, 0]
            shifts[axis] = direction
            neighbor_psi = torch.roll(
                psi, shifts=tuple(shifts), dims=(0, 1, 2))
            neighbor_chi = torch.roll(
                chi, shifts=tuple(shifts), dims=(0, 1, 2))
            psi_link = incumbent.link_phase(psi, neighbor_psi, "phase")
            chi_link = incumbent.link_phase(chi, neighbor_chi, "phase")
            result = result+torch.real(1j*psi_link*torch.conj(chi_link))
    return result


def grade_one_form(left: torch.Tensor, right: torch.Tensor,
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
    psi_covariant_dot = psi_dot-1j*centered_clock*psi
    phase_velocity = torch.imag(torch.sum(
        torch.conj(psi)*psi_covariant_dot, dim=-1))
    return centered_clock*torch.sum(live_grade(psi, chi)*phase_velocity)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    return (incumbent.interval(left, right, dt)
            + dt*GRADE_SCALE*grade_one_form(left, right, dt))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
