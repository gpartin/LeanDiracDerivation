"""Rest-null transverse transport of the native Psi--chi area ray.

Let w=(chi cross Psi)/chi0 and D_t w be its material rate under the common
native translation velocity.  The polynomial Gram determinant

    |w|^2 |D_t w|^2 - |w^dagger D_t w|^2

is nonnegative by Cauchy--Schwarz and vanishes exactly when the material rate
is parallel to w.  It therefore ignores amplitude and common phase motion and
acts only on changes of the live oriented-area ray.  It is also zero for exact
common translation (D_t w=0).  No prepared ray, projector, direction, or
Dirac object occurs in the action.  Division by chi0^2 supplies the native
C19 geometric normalization without a fitted coefficient.
"""

from __future__ import annotations

import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import exact_translation_multichannel_prong_gate as common
import wake_stabilized_completed_action as incumbent


SIDE = common.SIDE


def oriented_area(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(chi, psi, dim=-1)/rank.R


def transverse_area_kinetic(w: torch.Tensor,
                            material_w: torch.Tensor) -> torch.Tensor:
    rho = common.squared_norm(w)
    rate_norm = common.squared_norm(material_w)
    overlap = torch.sum(torch.conj(w)*material_w, dim=-1)
    gram_determinant = rho*rate_norm-torch.abs(overlap)**2
    # Keep the smooth polynomial exactly as derived.  Cauchy--Schwarz makes
    # it nonnegative analytically; clipping at its rest-orbit zero would make
    # the Hessian of the discrete action undefined.
    return torch.sum(gram_determinant)/(2*rank.R**2)


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left, SIDE)
    right_psi, right_chi = local.unpack_tensor(right, SIDE)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")

    w_left = oriented_area(left_psi, left_chi)
    w_right = oriented_area(right_psi, right_chi)
    w_mid = (w_left+w_right)/2
    w_dot = (w_right-w_left)/dt
    material_w = w_dot+common.advect_same_components(w_mid, velocity)
    return (incumbent.interval(left, right, dt)
            + dt*transverse_area_kinetic(w_mid, material_w))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
