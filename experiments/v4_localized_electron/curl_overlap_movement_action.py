"""Rest-null movement coupling for the complete live Psi--chi frame.

Let ``s = chi^dagger Psi / chi0`` and ``w = chi wedge Psi / chi0``.  The
canonical area action transports ``w`` but not the parallel coordinate ``s``.
The first moving departure contains both channels.  A parity-even local
completion is obtained by pairing two polar C19 vectors: ``grad(s)`` and the
curl of ``w``.  This module adds the single positive square

  1/2 |(D_t - i omega0) curl(w) + grad(s)|^2.

The exact centered-time rest frequency is used for ``omega0``.  Hence the new
term and its first variation vanish on the rotating rest carrier, but its
Hessian is active for motion.  It contains no carrier labels, chosen axis,
copy projector, or Dirac object and varies reciprocally through Psi and chi.
"""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("LFM_RELATIVE_PRONG_WEIGHT", "0")

import canonical_noncoplanar_bifurcation_audit as native_time
import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def overlap(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.conj(chi)*psi, dim=-1, keepdim=True)/rank.R


def area(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(chi, psi, dim=-1)/rank.R


def curl(value: torch.Tensor) -> torch.Tensor:
    gradient = common.complex_gradient(value)
    return torch.stack((
        gradient[..., 2, 1]-gradient[..., 1, 2],
        gradient[..., 0, 2]-gradient[..., 2, 0],
        gradient[..., 1, 0]-gradient[..., 0, 1],
    ), dim=-1)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")

    s_mid = (overlap(left_psi, left_chi)
             + overlap(right_psi, right_chi))/2
    grad_s = common.complex_gradient(s_mid)[..., 0, :]

    curl_left = curl(area(left_psi, left_chi))
    curl_right = curl(area(right_psi, right_chi))
    curl_mid = (curl_left+curl_right)/2
    curl_dot = (curl_right-curl_left)/dt
    material_curl = curl_dot + common.advect_same_components(
        curl_mid, velocity)

    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    centered_clock = 2*np.tan(theta/2)/dt
    residual = material_curl-1j*centered_clock*curl_mid+grad_s
    completion = torch.sum(common.squared_norm(residual))/2
    return incumbent.interval(left, right, dt)+dt*completion


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
