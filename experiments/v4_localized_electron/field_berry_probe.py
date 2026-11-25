"""Test the remaining lowest-degree local Psi and chi Berry one-forms."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares

import common_gyroscopic_probe as prior


HERE = Path(__file__).resolve().parent


def set_chi_inertia(value):
    prior.bond.B = value
    prior.bond.local.B = value
    prior.bond.local.common.B = value
    prior.bond.local.common.base.B = value


TRIANGLE_PAIRS = tuple(
    pair
    for first, second in prior.bond.local.common.base.PLAQUETTES
    for pair in ((first, second), (second, first)))
TRIANGLE_NORMALIZATION = float(len(TRIANGLE_PAIRS))


def shift_adjoint(field, offset):
    reverse = tuple(-value for value in offset)
    return prior.bond.local.common.base.shift(field, reverse) - field


def triangle_density(area, scale=None):
    """Parity-even chirality of the elementary C19 face-edge triangles."""
    density = np.zeros(area.shape[:-1])
    for first, second in TRIANGLE_PAIRS:
        u = prior.bond.local.common.base.shift(area, first) - area
        v = prior.bond.local.common.base.shift(area, second) - area
        spatial_area = np.cross(np.asarray(first), np.asarray(second))
        density += np.real(np.sum(
            u.conj() * np.cross(v, spatial_area), axis=-1))
    if scale is None:
        scale = prior.R**4
    return density / (TRIANGLE_NORMALIZATION * scale)


def triangle_density_gradient(area, multiplier, scale=None):
    """Real gradient of sum_i multiplier_i * triangle_density_i."""
    gradient = np.zeros_like(area)
    if scale is None:
        scale = prior.R**4
    normalization = TRIANGLE_NORMALIZATION * scale
    for first, second in TRIANGLE_PAIRS:
        u = prior.bond.local.common.base.shift(area, first) - area
        v = prior.bond.local.common.base.shift(area, second) - area
        spatial_area = np.cross(np.asarray(first), np.asarray(second))
        grad_u = multiplier[..., None] * np.cross(v, spatial_area)
        grad_v = -multiplier[..., None] * np.cross(u, spatial_area)
        gradient += shift_adjoint(grad_u, first)
        gradient += shift_adjoint(grad_v, second)
    return gradient / normalization


def directional_area_gradient(area, strength):
    """Gradient of the positive cubic longitudinal bond energy.

    The sum contains every C19 face and edge direction, so it selects no
    preferred axis while coupling internal oriented area to physical bond
    geometry.
    """
    result = np.zeros_like(area)
    for offset, weight in prior.bond.local.common.base.DIRECTIONS:
        direction = np.asarray(offset, dtype=float)
        direction /= np.linalg.norm(direction)
        difference = area - prior.bond.local.common.base.shift(area, offset)
        longitudinal = np.sum(difference * direction, axis=-1)[..., None]
        result += weight * longitudinal * direction
    return 2 * strength / prior.R**2 * result


def normalized_cross_covariance_bond_gradients(psi, chi, strength):
    """Gradient of a positive C19 bond energy for the full joint relation.

    X=chi Psi^dagger and Z=R X/sqrt(R^4+||X||_F^2).  The energy is

        strength/(2 sum_w) <Z,D19 Z>.

    It is real, nonnegative for strength >= 0, reciprocal, cubic symmetric,
    and contains no prepared carrier data.  Returned values are dU/dPsi and
    dU/dChi in the real-pairing convention used throughout this research code.
    """
    relation = chi[..., :, None] * psi.conj()[..., None, :]
    density = np.sum(abs(relation)**2, axis=(-2, -1), keepdims=True)
    denominator = prior.R**4 + density
    h = prior.R / np.sqrt(denominator)
    h_prime = -prior.R / (2 * denominator**1.5)
    normalized = h * relation
    momentum_normalized = (
        strength / prior.bond.C19_WEIGHT_SUM
        * prior.bond.local.common.base.c19_dirichlet(normalized))
    radial = np.real(np.sum(
        relation.conj() * momentum_normalized,
        axis=(-2, -1), keepdims=True))
    momentum_relation = (h * momentum_normalized
                         + 2 * h_prime * radial * relation)
    gradient_psi = np.einsum(
        "...ab,...a->...b", momentum_relation.conj(), chi)
    gradient_chi = np.einsum(
        "...ab,...b->...a", momentum_relation, psi)
    return gradient_psi, gradient_chi


def normalized_area_bond_derivatives(psi, chi, velocity_psi, velocity_chi,
                                     coefficient, relative):
    """Pull back Im[Z^dag D19 Zdot] through Z=R W/sqrt(R^4+|W|^2).

    This is one C19/cubic-symmetric real one-form.  The returned tuple is
    (dL/dPsiDot, dL/dChiDot, dL/dPsi, dL/dChi); no prepared corner data enter
    the construction.
    """
    area = np.cross(chi, psi)
    if relative:
        area_velocity = (np.cross(chi, velocity_psi)
                         - np.cross(velocity_chi, psi))
    else:
        area_velocity = (np.cross(chi, velocity_psi)
                         + np.cross(velocity_chi, psi))
    density = prior.bond.local.common.base.norm2(area)[..., None]
    denominator = prior.R**4 + density
    h = prior.R / np.sqrt(denominator)
    h_prime = -prior.R / (2 * denominator**1.5)
    h_second = 3 * prior.R / (4 * denominator**2.5)

    def pullback(value):
        radial = np.real(np.sum(area.conj() * value, axis=-1))[..., None]
        return h * value + 2 * h_prime * radial * area

    normalized_area = h * area
    normalized_velocity = pullback(area_velocity)
    scale = coefficient / prior.bond.C19_WEIGHT_SUM
    momentum_normalized = (1j * scale
                           * prior.bond.local.common.base.c19_dirichlet(
                               normalized_area))
    position_normalized = (-1j * scale
                           * prior.bond.local.common.base.c19_dirichlet(
                               normalized_velocity))
    momentum_area = pullback(momentum_normalized)

    radial_velocity = np.real(np.sum(
        area.conj() * area_velocity, axis=-1))[..., None]
    momentum_on_velocity = np.real(np.sum(
        momentum_normalized.conj() * area_velocity,
        axis=-1))[..., None]
    momentum_on_area = np.real(np.sum(
        momentum_normalized.conj() * area, axis=-1))[..., None]
    position_area = (
        pullback(position_normalized)
        + 2 * h_prime * momentum_on_velocity * area
        + 4 * h_second * radial_velocity * momentum_on_area * area
        + 2 * h_prime * momentum_on_area * area_velocity
        + 2 * h_prime * radial_velocity * momentum_normalized)

    momentum_psi = np.cross(momentum_area, chi.conj())
    if relative:
        momentum_chi = np.cross(momentum_area, psi.conj())
        position_psi = (np.cross(position_area, chi.conj())
                        - np.cross(momentum_area, velocity_chi.conj()))
    else:
        momentum_chi = np.cross(psi.conj(), momentum_area)
        position_psi = (np.cross(position_area, chi.conj())
                        + np.cross(momentum_area, velocity_chi.conj()))
    position_chi = (np.cross(psi.conj(), position_area)
                    + np.cross(velocity_psi.conj(), momentum_area))
    return momentum_psi, momentum_chi, position_psi, position_chi


def normalized_area_wave_derivatives(psi, chi, velocity_psi, velocity_chi,
                                     coefficient):
    """Derivatives of the positive C19 wave action for normalized area.

    Delta L = eta/(2 sum w) [<Zdot,D19 Zdot> - <Z,D19^2 Z>],
    Z = R (chi x psi)/sqrt(R^4+|chi x psi|^2).
    Its Hamiltonian contribution is positive for eta >= 0 and its kinetic and
    restoring pieces have the same native C19 wave speed.
    """
    area = np.cross(chi, psi)
    area_velocity = (np.cross(chi, velocity_psi)
                     + np.cross(velocity_chi, psi))
    density = prior.bond.local.common.base.norm2(area)[..., None]
    denominator = prior.R**4 + density
    h = prior.R / np.sqrt(denominator)
    h_prime = -prior.R / (2 * denominator**1.5)
    h_second = 3 * prior.R / (4 * denominator**2.5)

    def pullback(value):
        radial = np.real(np.sum(area.conj() * value, axis=-1))[..., None]
        return h * value + 2 * h_prime * radial * area

    def second_pullback(momentum, velocity):
        radial_velocity = np.real(np.sum(
            area.conj() * velocity, axis=-1))[..., None]
        momentum_on_velocity = np.real(np.sum(
            momentum.conj() * velocity, axis=-1))[..., None]
        momentum_on_area = np.real(np.sum(
            momentum.conj() * area, axis=-1))[..., None]
        return (
            2 * h_prime * momentum_on_velocity * area
            + 4 * h_second * radial_velocity * momentum_on_area * area
            + 2 * h_prime * momentum_on_area * velocity
            + 2 * h_prime * radial_velocity * momentum)

    normalized_area = h * area
    normalized_velocity = pullback(area_velocity)
    d_velocity = prior.bond.local.common.base.c19_dirichlet(
        normalized_velocity)
    d_area = prior.bond.local.common.base.c19_dirichlet(normalized_area)
    d2_area = prior.bond.local.common.base.c19_dirichlet(d_area)
    scale = coefficient / prior.bond.C19_WEIGHT_SUM
    momentum_normalized = scale * d_velocity
    momentum_area = pullback(momentum_normalized)
    position_area = (second_pullback(momentum_normalized, area_velocity)
                     - pullback(scale * d2_area))

    momentum_psi = np.cross(momentum_area, chi.conj())
    momentum_chi = np.cross(psi.conj(), momentum_area)
    position_psi = (np.cross(position_area, chi.conj())
                    + np.cross(momentum_area, velocity_chi.conj()))
    position_chi = (np.cross(psi.conj(), position_area)
                    + np.cross(velocity_psi.conj(), momentum_area))
    return momentum_psi, momentum_chi, position_psi, position_chi


def all_derivatives(left_psi, left_chi, right_psi, right_chi, dt,
                    relative_local, relative_bond, common_gamma,
                    psi_berry, chi_berry,
                    psi_density_common, chi_density_common,
                    psi_berry_rho_psi=0.0, psi_berry_rho_chi=0.0,
                    chi_berry_rho_psi=0.0, chi_berry_rho_chi=0.0,
                    psi_density_relative=0.0, chi_density_relative=0.0,
                    triangle_common=0.0,
                    triangle_psi=0.0, triangle_chi=0.0,
                    parallel_kinetic=0.0,
                    psi_gradient_kinetic=0.0,
                    chi_gradient_kinetic=0.0,
                    area_gradient_kinetic=0.0,
                    chi_berry_density_difference_sq=0.0,
                    psi_bond_berry=0.0, chi_bond_berry=0.0,
                    area_local_kinetic=0.0,
                    overlap_common_berry=0.0,
                    overlap_relative_berry=0.0,
                    area_relative_gradient_kinetic=0.0,
                    overlap_bond_common_berry=0.0,
                    overlap_bond_relative_berry=0.0,
                    area_plaquette_relative_berry=0.0,
                    overlap_common_kinetic=0.0,
                    psi_orientation_kinetic=0.0,
                    chi_orientation_kinetic=0.0,
                    chi_berry_area_weight=0.0,
                    psi_berry_area_weight=0.0,
                    normalized_area_common=0.0,
                    normalized_area_relative=0.0,
                    normalized_area_bond_common=0.0,
                    normalized_area_bond_relative=0.0,
                    normalized_area_wave=0.0):
    values = list(prior.all_derivatives(
        left_psi, left_chi, right_psi, right_chi, dt,
        relative_local, relative_bond, common_gamma))
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    values[0] += psi_berry * 1j * psi
    values[1] += chi_berry * 1j * chi
    values[2] += -psi_berry * 1j * velocity_psi
    values[3] += -chi_berry * 1j * velocity_chi
    density_psi = prior.bond.local.common.base.norm2(psi)[..., None]
    density_chi = prior.bond.local.common.base.norm2(chi)[..., None]
    if psi_berry_rho_psi != 0 or psi_berry_rho_chi != 0:
        weight = ((psi_berry_rho_psi * density_psi
                   + psi_berry_rho_chi * density_chi) / prior.R**2)
        berry_density = np.imag(np.sum(
            psi.conj() * velocity_psi, axis=-1))[..., None]
        values[0] += 1j * weight * psi
        values[2] += (-1j * weight * velocity_psi
                      + 2 * psi_berry_rho_psi / prior.R**2
                      * berry_density * psi)
        values[3] += (2 * psi_berry_rho_chi / prior.R**2
                      * berry_density * chi)
    if chi_berry_rho_psi != 0 or chi_berry_rho_chi != 0:
        weight = ((chi_berry_rho_psi * density_psi
                   + chi_berry_rho_chi * density_chi) / prior.R**2)
        berry_density = np.imag(np.sum(
            chi.conj() * velocity_chi, axis=-1))[..., None]
        values[1] += 1j * weight * chi
        values[2] += (2 * chi_berry_rho_psi / prior.R**2
                      * berry_density * psi)
        values[3] += (-1j * weight * velocity_chi
                      + 2 * chi_berry_rho_chi / prior.R**2
                      * berry_density * chi)
    if psi_density_common != 0 or chi_density_common != 0:
        unit_common = prior.common_derivatives(
            left_psi, left_chi, right_psi, right_chi, dt, 1.0)
        ordinary = (velocity_psi, prior.bond.B * velocity_chi,
                    np.zeros_like(velocity_psi), np.zeros_like(velocity_chi))
        increments = tuple(a - b for a, b in zip(unit_common, ordinary))
        weight = ((psi_density_common * density_psi
                   + chi_density_common * density_chi) / prior.R**2)
        area = np.cross(chi, psi)
        area_velocity = (np.cross(chi, velocity_psi)
                         + np.cross(velocity_chi, psi))
        common_density = np.imag(np.sum(
            area.conj() * area_velocity, axis=-1))[..., None] / prior.R**2
        values = [value + weight * increment
                  for value, increment in zip(values, increments)]
        values[2] += (2 * psi_density_common / prior.R**2
                      * common_density * psi)
        values[3] += (2 * chi_density_common / prior.R**2
                      * common_density * chi)
    if psi_density_relative != 0 or chi_density_relative != 0:
        unit_relative = prior.bond.local.gyro_derivatives(
            left_psi, left_chi, right_psi, right_chi, dt, 1.0)
        ordinary = (velocity_psi, prior.bond.B * velocity_chi,
                    np.zeros_like(velocity_psi), np.zeros_like(velocity_chi))
        increments = tuple(a - b for a, b in zip(unit_relative, ordinary))
        weight = ((psi_density_relative * density_psi
                   + chi_density_relative * density_chi) / prior.R**2)
        area = np.cross(chi, psi)
        relative_velocity = (np.cross(chi, velocity_psi)
                             - np.cross(velocity_chi, psi))
        relative_density = np.imag(np.sum(
            area.conj() * relative_velocity, axis=-1))[..., None] / prior.R**2
        values = [value + weight * increment
                  for value, increment in zip(values, increments)]
        values[2] += (2 * psi_density_relative / prior.R**2
                      * relative_density * psi)
        values[3] += (2 * chi_density_relative / prior.R**2
                      * relative_density * chi)
    if triangle_common != 0:
        area = np.cross(chi, psi)
        area_velocity = (np.cross(chi, velocity_psi)
                         + np.cross(velocity_chi, psi))
        chirality = triangle_density(area)
        common_density = np.imag(np.sum(
            area.conj() * area_velocity, axis=-1)) / prior.R**2
        momentum_area = (1j * triangle_common / prior.R**2
                         * chirality[..., None]**2 * area)
        position_area = triangle_common * (
            -1j / prior.R**2 * chirality[..., None]**2 * area_velocity
            + triangle_density_gradient(
                area, 2 * chirality * common_density))
        values[0] += np.cross(momentum_area, chi.conj())
        values[1] += np.cross(psi.conj(), momentum_area)
        values[2] += (np.cross(position_area, chi.conj())
                      + np.cross(momentum_area, velocity_chi.conj()))
        values[3] += (np.cross(psi.conj(), position_area)
                      + np.cross(velocity_psi.conj(), momentum_area))
    for coefficient, field_value, field_velocity, momentum_index, position_index in (
            (triangle_psi, psi, velocity_psi, 0, 2),
            (triangle_chi, chi, velocity_chi, 1, 3)):
        if coefficient == 0:
            continue
        chirality = triangle_density(field_value, prior.R**2)
        berry_density = np.imag(np.sum(
            field_value.conj() * field_velocity, axis=-1)) / prior.R**2
        values[momentum_index] += (
            1j * coefficient / prior.R**2
            * chirality[..., None] * field_value)
        values[position_index] += coefficient * (
            -1j / prior.R**2 * chirality[..., None] * field_velocity
            + triangle_density_gradient(
                field_value, berry_density, prior.R**2))
    if parallel_kinetic != 0:
        relative_parallel = (np.sum(chi.conj() * velocity_psi, axis=-1)
                             - np.sum(velocity_chi.conj() * psi, axis=-1))
        relative_parallel = relative_parallel[..., None]
        scale = parallel_kinetic / prior.R**2
        values[0] += scale * relative_parallel * chi
        values[1] -= scale * relative_parallel.conj() * psi
        values[2] -= scale * relative_parallel * velocity_chi
        values[3] += scale * relative_parallel.conj() * velocity_psi
    c19_weight_sum = prior.bond.C19_WEIGHT_SUM
    if psi_gradient_kinetic != 0:
        values[0] += (psi_gradient_kinetic / c19_weight_sum
                      * prior.bond.local.common.base.c19_dirichlet(
                          velocity_psi))
    if chi_gradient_kinetic != 0:
        values[1] += (prior.bond.B * chi_gradient_kinetic / c19_weight_sum
                      * prior.bond.local.common.base.c19_dirichlet(
                          velocity_chi))
    if area_gradient_kinetic != 0:
        area_velocity = (np.cross(chi, velocity_psi)
                         + np.cross(velocity_chi, psi))
        area_momentum = (
            area_gradient_kinetic
            / (prior.R**2 * c19_weight_sum)
            * prior.bond.local.common.base.c19_dirichlet(area_velocity))
        values[0] += np.cross(area_momentum, chi.conj())
        values[1] += np.cross(psi.conj(), area_momentum)
        values[2] += np.cross(area_momentum, velocity_chi.conj())
        values[3] += np.cross(velocity_psi.conj(), area_momentum)
    if chi_berry_density_difference_sq != 0:
        difference = density_psi - density_chi
        weight = (chi_berry_density_difference_sq * difference**2
                  / prior.R**4)
        berry_density = np.imag(np.sum(
            chi.conj() * velocity_chi, axis=-1))[..., None]
        response = (4 * chi_berry_density_difference_sq * difference
                    * berry_density / prior.R**4)
        values[1] += 1j * weight * chi
        values[2] += response * psi
        values[3] += -1j * weight * velocity_chi - response * chi
    if psi_bond_berry != 0:
        scale = psi_bond_berry / c19_weight_sum
        values[0] += (1j * scale
                      * prior.bond.local.common.base.c19_dirichlet(psi))
        values[2] -= (1j * scale
                      * prior.bond.local.common.base.c19_dirichlet(
                          velocity_psi))
    if chi_bond_berry != 0:
        scale = chi_bond_berry / c19_weight_sum
        values[1] += (1j * scale
                      * prior.bond.local.common.base.c19_dirichlet(chi))
        values[3] -= (1j * scale
                      * prior.bond.local.common.base.c19_dirichlet(
                          velocity_chi))
    if area_local_kinetic != 0:
        area_velocity = (np.cross(chi, velocity_psi)
                         + np.cross(velocity_chi, psi))
        area_momentum = area_local_kinetic / prior.R**2 * area_velocity
        values[0] += np.cross(area_momentum, chi.conj())
        values[1] += np.cross(psi.conj(), area_momentum)
        values[2] += np.cross(area_momentum, velocity_chi.conj())
        values[3] += np.cross(velocity_psi.conj(), area_momentum)
    overlap = np.sum(chi.conj() * psi, axis=-1)[..., None]
    if overlap_common_berry != 0:
        overlap_velocity = (
            np.sum(chi.conj() * velocity_psi, axis=-1)
            + np.sum(velocity_chi.conj() * psi, axis=-1))[..., None]
        momentum_overlap = 1j * overlap_common_berry * overlap
        position_overlap = -1j * overlap_common_berry * overlap_velocity
        values[0] += chi * momentum_overlap
        values[1] += momentum_overlap.conj() * psi
        values[2] += (chi * position_overlap
                      + velocity_chi * momentum_overlap)
        values[3] += (position_overlap.conj() * psi
                      + momentum_overlap.conj() * velocity_psi)
    if overlap_relative_berry != 0:
        relative_overlap = (
            np.sum(chi.conj() * velocity_psi, axis=-1)
            - np.sum(velocity_chi.conj() * psi, axis=-1))[..., None]
        momentum_overlap = 1j * overlap_relative_berry * overlap
        position_overlap = -1j * overlap_relative_berry * relative_overlap
        values[0] += chi * momentum_overlap
        values[1] -= momentum_overlap.conj() * psi
        values[2] += (chi * position_overlap
                      - velocity_chi * momentum_overlap)
        values[3] += (position_overlap.conj() * psi
                      + momentum_overlap.conj() * velocity_psi)
    if area_relative_gradient_kinetic != 0:
        relative_area_velocity = (np.cross(chi, velocity_psi)
                                  - np.cross(velocity_chi, psi))
        relative_area_momentum = (
            area_relative_gradient_kinetic
            / (prior.R**2 * c19_weight_sum)
            * prior.bond.local.common.base.c19_dirichlet(
                relative_area_velocity))
        values[0] += np.cross(relative_area_momentum, chi.conj())
        values[1] += np.cross(relative_area_momentum, psi.conj())
        values[2] += np.cross(velocity_chi.conj(), relative_area_momentum)
        values[3] += np.cross(velocity_psi.conj(), relative_area_momentum)
    if overlap_bond_common_berry != 0:
        overlap_velocity = (
            np.sum(chi.conj() * velocity_psi, axis=-1)
            + np.sum(velocity_chi.conj() * psi, axis=-1))[..., None]
        momentum_overlap = (
            1j * overlap_bond_common_berry / c19_weight_sum
            * prior.bond.local.common.base.c19_dirichlet(overlap))
        position_overlap = (
            -1j * overlap_bond_common_berry / c19_weight_sum
            * prior.bond.local.common.base.c19_dirichlet(overlap_velocity))
        values[0] += chi * momentum_overlap
        values[1] += momentum_overlap.conj() * psi
        values[2] += (chi * position_overlap
                      + velocity_chi * momentum_overlap)
        values[3] += (position_overlap.conj() * psi
                      + momentum_overlap.conj() * velocity_psi)
    if overlap_bond_relative_berry != 0:
        relative_overlap = (
            np.sum(chi.conj() * velocity_psi, axis=-1)
            - np.sum(velocity_chi.conj() * psi, axis=-1))[..., None]
        momentum_overlap = (
            1j * overlap_bond_relative_berry / c19_weight_sum
            * prior.bond.local.common.base.c19_dirichlet(overlap))
        position_overlap = (
            -1j * overlap_bond_relative_berry / c19_weight_sum
            * prior.bond.local.common.base.c19_dirichlet(relative_overlap))
        values[0] += chi * momentum_overlap
        values[1] -= momentum_overlap.conj() * psi
        values[2] += (chi * position_overlap
                      - velocity_chi * momentum_overlap)
        values[3] += (position_overlap.conj() * psi
                      + momentum_overlap.conj() * velocity_psi)
    if area_plaquette_relative_berry != 0:
        area = np.cross(chi, psi)
        relative_area_velocity = (np.cross(chi, velocity_psi)
                                  - np.cross(velocity_chi, psi))
        filtered_area = prior.bond.local.common.base.plaquette_dirichlet(area)
        filtered_relative = (
            prior.bond.local.common.base.plaquette_dirichlet(
                relative_area_velocity))
        scale = (area_plaquette_relative_berry
                 / (prior.R**2
                    * len(prior.bond.local.common.base.PLAQUETTES)))
        values[0] += scale * 1j * np.cross(filtered_area, chi.conj())
        values[1] += scale * 1j * np.cross(filtered_area, psi.conj())
        values[2] += scale * 1j * (
            np.cross(velocity_chi.conj(), filtered_area)
            - np.cross(filtered_relative, chi.conj()))
        values[3] += scale * 1j * (
            np.cross(velocity_psi.conj(), filtered_area)
            - np.cross(psi.conj(), filtered_relative))
    if overlap_common_kinetic != 0:
        overlap_velocity = (
            np.sum(chi.conj() * velocity_psi, axis=-1)
            + np.sum(velocity_chi.conj() * psi, axis=-1))[..., None]
        scale = overlap_common_kinetic / prior.R**2
        values[0] += scale * overlap_velocity * chi
        values[1] += scale * overlap_velocity.conj() * psi
        values[2] += scale * overlap_velocity * velocity_chi
        values[3] += scale * overlap_velocity.conj() * velocity_psi
    if psi_orientation_kinetic != 0:
        oriented_velocity = np.cross(chi, velocity_psi)
        oriented_momentum = (psi_orientation_kinetic / prior.R**2
                             * oriented_velocity)
        values[0] += np.cross(oriented_momentum, chi.conj())
        values[3] += np.cross(velocity_psi.conj(), oriented_momentum)
    if chi_orientation_kinetic != 0:
        oriented_velocity = np.cross(velocity_chi, psi)
        oriented_momentum = (chi_orientation_kinetic / prior.R**2
                             * oriented_velocity)
        values[1] -= np.cross(oriented_momentum, psi.conj())
        values[2] -= np.cross(velocity_chi.conj(), oriented_momentum)
    if chi_berry_area_weight != 0:
        area = np.cross(chi, psi)
        area_density = prior.bond.local.common.base.norm2(area)[..., None]
        overlap_scalar = np.sum(chi.conj() * psi, axis=-1)[..., None]
        berry_density = np.imag(np.sum(
            chi.conj() * velocity_chi, axis=-1))[..., None]
        scale = chi_berry_area_weight / prior.R**4
        weight = scale * area_density
        values[1] += 1j * weight * chi
        values[2] += 2 * scale * berry_density * (
            density_chi * psi - overlap_scalar * chi)
        values[3] += (-1j * weight * velocity_chi
                      + 2 * scale * berry_density * (
                          density_psi * chi
                          - overlap_scalar.conj() * psi))
    if psi_berry_area_weight != 0:
        area = np.cross(chi, psi)
        area_density = prior.bond.local.common.base.norm2(area)[..., None]
        overlap_scalar = np.sum(chi.conj() * psi, axis=-1)[..., None]
        berry_density = np.imag(np.sum(
            psi.conj() * velocity_psi, axis=-1))[..., None]
        scale = psi_berry_area_weight / prior.R**4
        weight = scale * area_density
        values[0] += 1j * weight * psi
        values[2] += (-1j * weight * velocity_psi
                      + 2 * scale * berry_density * (
                          density_chi * psi - overlap_scalar * chi))
        values[3] += 2 * scale * berry_density * (
            density_psi * chi - overlap_scalar.conj() * psi)
    area = np.cross(chi, psi)
    area_density = prior.bond.local.common.base.norm2(area)[..., None]
    denominator = prior.R**4 + area_density
    normalized_weight = prior.R**2 / denominator
    normalized_weight_derivative = -prior.R**2 / denominator**2
    for coefficient, area_velocity, relative in (
            (normalized_area_common,
             np.cross(chi, velocity_psi) + np.cross(velocity_chi, psi),
             False),
            (normalized_area_relative,
             np.cross(chi, velocity_psi) - np.cross(velocity_chi, psi),
             True)):
        if coefficient == 0:
            continue
        berry_density = np.imag(np.sum(
            area.conj() * area_velocity, axis=-1))[..., None]
        momentum_area = 1j * coefficient * normalized_weight * area
        position_area = coefficient * (
            -1j * normalized_weight * area_velocity
            + 2 * normalized_weight_derivative * berry_density * area)
        values[0] += np.cross(momentum_area, chi.conj())
        if relative:
            values[1] += np.cross(momentum_area, psi.conj())
            values[2] += (np.cross(position_area, chi.conj())
                          - np.cross(momentum_area, velocity_chi.conj()))
        else:
            values[1] += np.cross(psi.conj(), momentum_area)
            values[2] += (np.cross(position_area, chi.conj())
                          + np.cross(momentum_area, velocity_chi.conj()))
        values[3] += (np.cross(psi.conj(), position_area)
                      + np.cross(velocity_psi.conj(), momentum_area))
    for coefficient, relative in (
            (normalized_area_bond_common, False),
            (normalized_area_bond_relative, True)):
        if coefficient == 0:
            continue
        increments = normalized_area_bond_derivatives(
            psi, chi, velocity_psi, velocity_chi, coefficient, relative)
        values = [value + increment
                  for value, increment in zip(values, increments)]
    if normalized_area_wave != 0:
        increments = normalized_area_wave_derivatives(
            psi, chi, velocity_psi, velocity_chi, normalized_area_wave)
        values = [value + increment
                  for value, increment in zip(values, increments)]
    return tuple(values)


def del_residual(previous, current, following, side, dt,
                 relative_local, relative_bond, common_gamma,
                 psi_berry, chi_berry,
                 psi_density_common, chi_density_common,
                 psi_berry_rho_psi=0.0, psi_berry_rho_chi=0.0,
                 chi_berry_rho_psi=0.0, chi_berry_rho_chi=0.0,
                 psi_density_relative=0.0, chi_density_relative=0.0,
                 triangle_common=0.0,
                 triangle_psi=0.0, triangle_chi=0.0,
                 parallel_kinetic=0.0,
                 psi_gradient_kinetic=0.0,
                 chi_gradient_kinetic=0.0,
                 area_gradient_kinetic=0.0,
                 parallel_potential=0.0,
                 psi_gradient_potential=0.0,
                 chi_gradient_potential=0.0,
                 chi_berry_density_difference_sq=0.0,
                 chi_radial_delta=0.0,
                 total_density_stiffness_delta=0.0,
                 psi_bond_berry=0.0, chi_bond_berry=0.0,
                 area_local_kinetic=0.0,
                 overlap_common_berry=0.0,
                 overlap_relative_berry=0.0,
                 area_relative_gradient_kinetic=0.0,
                 overlap_bond_common_berry=0.0,
                 overlap_bond_relative_berry=0.0,
                 area_plaquette_relative_berry=0.0,
                 overlap_common_kinetic=0.0,
                 psi_orientation_kinetic=0.0,
                 chi_orientation_kinetic=0.0,
                 chi_berry_area_weight=0.0,
                 psi_berry_area_weight=0.0,
                 normalized_area_common=0.0,
                 normalized_area_relative=0.0,
                 normalized_area_bond_common=0.0,
                 normalized_area_bond_relative=0.0,
                 normalized_area_wave=0.0,
                 directional_area_potential=0.0,
                 cross_covariance_bond_potential=0.0):
    pm_psi, pm_chi = prior.bond.local.common.unpack_raw(previous, side)
    q_psi, q_chi = prior.bond.local.common.unpack_raw(current, side)
    pp_psi, pp_chi = prior.bond.local.common.unpack_raw(following, side)
    minus = all_derivatives(pm_psi, pm_chi, q_psi, q_chi, dt,
                            relative_local, relative_bond, common_gamma,
                            psi_berry, chi_berry,
                            psi_density_common, chi_density_common,
                            psi_berry_rho_psi, psi_berry_rho_chi,
                            chi_berry_rho_psi, chi_berry_rho_chi,
                            psi_density_relative, chi_density_relative,
                            triangle_common, triangle_psi, triangle_chi,
                            parallel_kinetic, psi_gradient_kinetic,
                            chi_gradient_kinetic, area_gradient_kinetic,
                            chi_berry_density_difference_sq,
                            psi_bond_berry, chi_bond_berry,
                            area_local_kinetic, overlap_common_berry,
                            overlap_relative_berry,
                            area_relative_gradient_kinetic,
                            overlap_bond_common_berry,
                            overlap_bond_relative_berry,
                            area_plaquette_relative_berry,
                            overlap_common_kinetic,
                            psi_orientation_kinetic,
                            chi_orientation_kinetic,
                            chi_berry_area_weight,
                            psi_berry_area_weight,
                            normalized_area_common,
                            normalized_area_relative,
                            normalized_area_bond_common,
                            normalized_area_bond_relative,
                            normalized_area_wave)
    plus = all_derivatives(q_psi, q_chi, pp_psi, pp_chi, dt,
                           relative_local, relative_bond, common_gamma,
                           psi_berry, chi_berry,
                           psi_density_common, chi_density_common,
                           psi_berry_rho_psi, psi_berry_rho_chi,
                           chi_berry_rho_psi, chi_berry_rho_chi,
                           psi_density_relative, chi_density_relative,
                           triangle_common, triangle_psi, triangle_chi,
                           parallel_kinetic, psi_gradient_kinetic,
                           chi_gradient_kinetic, area_gradient_kinetic,
                           chi_berry_density_difference_sq,
                           psi_bond_berry, chi_bond_berry,
                           area_local_kinetic, overlap_common_berry,
                           overlap_relative_berry,
                           area_relative_gradient_kinetic,
                           overlap_bond_common_berry,
                           overlap_bond_relative_berry,
                           area_plaquette_relative_berry,
                           overlap_common_kinetic,
                           psi_orientation_kinetic,
                           chi_orientation_kinetic,
                           chi_berry_area_weight,
                           psi_berry_area_weight,
                           normalized_area_common,
                           normalized_area_relative,
                           normalized_area_bond_common,
                           normalized_area_bond_relative,
                           normalized_area_wave)
    force_psi, force_chi = prior.bond.local.common.potential_force(q_psi, q_chi)
    if directional_area_potential != 0:
        area_gradient = directional_area_gradient(
            np.cross(q_chi, q_psi), directional_area_potential)
        force_psi -= np.cross(area_gradient, q_chi.conj())
        force_chi -= np.cross(q_psi.conj(), area_gradient)
    if cross_covariance_bond_potential != 0:
        gradient_psi, gradient_chi = (
            normalized_cross_covariance_bond_gradients(
                q_psi, q_chi, cross_covariance_bond_potential))
        force_psi -= gradient_psi
        force_chi -= gradient_chi
    # The gradient kinetic term is admitted only as the time part of the
    # positive covariant higher-gradient action
    #   eta/(2 sum_w) [<Psi_dot,D19 Psi_dot> + <Psi,D19^2 Psi>]_H.
    # Its potential variation is therefore reciprocal and fixed by c=1.
    if len((relative_local, relative_bond, common_gamma, psi_berry, chi_berry,
            psi_density_common, chi_density_common,
            psi_berry_rho_psi, psi_berry_rho_chi,
            chi_berry_rho_psi, chi_berry_rho_chi,
            psi_density_relative, chi_density_relative,
            triangle_common, triangle_psi, triangle_chi,
            parallel_kinetic, psi_gradient_kinetic,
            chi_gradient_kinetic)) > 17 and psi_gradient_kinetic != 0:
        first = prior.bond.local.common.base.c19_dirichlet(q_psi)
        second = prior.bond.local.common.base.c19_dirichlet(first)
        force_psi -= (psi_gradient_kinetic / prior.bond.C19_WEIGHT_SUM
                      * second)
    if parallel_potential != 0:
        overlap = np.sum(q_chi.conj() * q_psi, axis=-1)[..., None]
        force_psi -= parallel_potential * overlap * q_chi
        force_chi -= parallel_potential * overlap.conj() * q_psi
    if psi_gradient_potential != 0:
        force_psi -= (
            psi_gradient_potential / prior.bond.C19_WEIGHT_SUM
            * prior.bond.local.common.base.c19_dirichlet(q_psi))
    if chi_gradient_potential != 0:
        force_chi -= (
            chi_gradient_potential / prior.bond.C19_WEIGHT_SUM
            * prior.bond.local.common.base.c19_dirichlet(q_chi))
    if chi_radial_delta != 0:
        rho_chi = prior.bond.local.common.base.norm2(q_chi)[..., None]
        force_chi -= chi_radial_delta * (rho_chi - prior.R**2) * q_chi
    if total_density_stiffness_delta != 0:
        rho_psi = prior.bond.local.common.base.norm2(q_psi)[..., None]
        rho_chi = prior.bond.local.common.base.norm2(q_chi)[..., None]
        mismatch = rho_psi + rho_chi - prior.R**2
        force_psi -= total_density_stiffness_delta * mismatch * q_psi
        force_chi -= total_density_stiffness_delta * mismatch * q_chi
    return prior.bond.local.common.pack_raw(
        minus[0] - plus[0] + dt * (minus[2] + plus[2]) / 2 + dt * force_psi,
        minus[1] - plus[1] + dt * (minus[3] + plus[3]) / 2 + dt * force_chi)


def solve_rest(side, dt, coefficients, guess):
    scale = np.sqrt(12 * side**3)

    def residual(parameters):
        layers = prior.bond.local.ansatz_layers(
            side, np.exp(parameters[0]), parameters[1], parameters[2])
        return del_residual(*layers, side, dt, *coefficients) / scale

    solution = least_squares(residual, guess, xtol=1e-13, ftol=1e-13,
                             gtol=1e-13, max_nfev=250)
    layers = prior.bond.local.ansatz_layers(
        side, np.exp(solution.x[0]), solution.x[1], solution.x[2])
    raw = del_residual(*layers, side, dt, *coefficients)
    return solution, layers, float(np.linalg.norm(raw) / np.sqrt(raw.size))


def jacobian(argument, layers, side, dt, coefficients, step=2e-6):
    dimension = layers[0].size
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        plus = [value.copy() for value in layers]
        minus = [value.copy() for value in layers]
        plus[argument][column] += step
        minus[argument][column] -= step
        result[:, column] = (
            del_residual(*plus, side, dt, *coefficients)
            - del_residual(*minus, side, dt, *coefficients)) / (2 * step)
    return result


def classify(side, dt, coefficients, guess, jacobian_step=2e-6):
    solution, layers, residual = solve_rest(side, dt, coefficients, guess)
    row = {
        "chi_inertia": prior.bond.B,
        "relative_local_gamma": coefficients[0],
        "relative_bond_gamma": coefficients[1],
        "common_gamma": coefficients[2],
        "psi_berry": coefficients[3],
        "chi_berry": coefficients[4],
        "psi_density_common": coefficients[5],
        "chi_density_common": coefficients[6],
        "psi_berry_rho_psi": coefficients[7] if len(coefficients) > 7 else 0.0,
        "psi_berry_rho_chi": coefficients[8] if len(coefficients) > 8 else 0.0,
        "chi_berry_rho_psi": coefficients[9] if len(coefficients) > 9 else 0.0,
        "chi_berry_rho_chi": coefficients[10] if len(coefficients) > 10 else 0.0,
        "psi_density_relative": coefficients[11] if len(coefficients) > 11 else 0.0,
        "chi_density_relative": coefficients[12] if len(coefficients) > 12 else 0.0,
        "triangle_common": coefficients[13] if len(coefficients) > 13 else 0.0,
        "triangle_psi": coefficients[14] if len(coefficients) > 14 else 0.0,
        "triangle_chi": coefficients[15] if len(coefficients) > 15 else 0.0,
        "parallel_kinetic": coefficients[16] if len(coefficients) > 16 else 0.0,
        "psi_gradient_kinetic": coefficients[17] if len(coefficients) > 17 else 0.0,
        "chi_gradient_kinetic": coefficients[18] if len(coefficients) > 18 else 0.0,
        "area_gradient_kinetic": coefficients[19] if len(coefficients) > 19 else 0.0,
        "parallel_potential": coefficients[20] if len(coefficients) > 20 else 0.0,
        "psi_gradient_potential": coefficients[21] if len(coefficients) > 21 else 0.0,
        "chi_gradient_potential": coefficients[22] if len(coefficients) > 22 else 0.0,
        "chi_berry_density_difference_sq": coefficients[23] if len(coefficients) > 23 else 0.0,
        "chi_radial_delta": coefficients[24] if len(coefficients) > 24 else 0.0,
        "total_density_stiffness_delta": coefficients[25] if len(coefficients) > 25 else 0.0,
        "psi_bond_berry": coefficients[26] if len(coefficients) > 26 else 0.0,
        "chi_bond_berry": coefficients[27] if len(coefficients) > 27 else 0.0,
        "area_local_kinetic": coefficients[28] if len(coefficients) > 28 else 0.0,
        "overlap_common_berry": coefficients[29] if len(coefficients) > 29 else 0.0,
        "overlap_relative_berry": coefficients[30] if len(coefficients) > 30 else 0.0,
        "area_relative_gradient_kinetic": coefficients[31] if len(coefficients) > 31 else 0.0,
        "overlap_bond_common_berry": coefficients[32] if len(coefficients) > 32 else 0.0,
        "overlap_bond_relative_berry": coefficients[33] if len(coefficients) > 33 else 0.0,
        "area_plaquette_relative_berry": coefficients[34] if len(coefficients) > 34 else 0.0,
        "overlap_common_kinetic": coefficients[35] if len(coefficients) > 35 else 0.0,
        "psi_orientation_kinetic": coefficients[36] if len(coefficients) > 36 else 0.0,
        "chi_orientation_kinetic": coefficients[37] if len(coefficients) > 37 else 0.0,
        "chi_berry_area_weight": coefficients[38] if len(coefficients) > 38 else 0.0,
        "psi_berry_area_weight": coefficients[39] if len(coefficients) > 39 else 0.0,
        "normalized_area_common": coefficients[40] if len(coefficients) > 40 else 0.0,
        "normalized_area_relative": coefficients[41] if len(coefficients) > 41 else 0.0,
        "normalized_area_bond_common": coefficients[42] if len(coefficients) > 42 else 0.0,
        "normalized_area_bond_relative": coefficients[43] if len(coefficients) > 43 else 0.0,
        "normalized_area_wave": coefficients[44] if len(coefficients) > 44 else 0.0,
        "directional_area_potential": coefficients[45] if len(coefficients) > 45 else 0.0,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "theta_psi": float(solution.x[1]),
        "theta_chi": float(solution.x[2]),
        "rest_del_rms_residual": residual,
        "rest_solver_success": bool(solution.success),
    }
    if residual > 1e-7:
        return row
    matrices = [jacobian(index, layers, side, dt, coefficients,
                         step=jacobian_step)
                for index in range(3)]
    rotation = prior.bond.local.common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    lhs = matrices[2] @ rotation
    forward = -np.linalg.solve(lhs, matrices[1])
    backward = -np.linalg.solve(lhs, matrices[0] @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([[forward, backward],
                          [identity, np.zeros_like(identity)]])
    values, vectors = eig(monodromy, check_finite=False)
    absolute = abs(values)
    order = np.argsort(absolute)[::-1]
    radius = float(absolute[order[0]])
    leading_raw = vectors[:forward.shape[0], order[0]]
    dpsi, dchi = prior.bond.local.common.unpack_raw(leading_raw, side)
    leading = prior.bond.local.common.base.pack(dpsi, dchi)
    leading /= np.linalg.norm(leading)
    current_psi, current_chi = prior.bond.local.common.unpack_raw(layers[1], side)
    _, corner, radial, phase = prior.bond.local.common.base.internal_mode_bases(
        current_psi, current_chi)
    mode = leading.reshape(side, side, side, 4, 3)
    psi_weight = np.sum(abs(mode[..., :2, :])**2, axis=(-2, -1))
    chi_weight = np.sum(abs(mode[..., 2:, :])**2, axis=(-2, -1))
    chi_components = np.sum(abs(mode[..., 2:, :])**2, axis=(0, 1, 2, 3))
    total = float(np.sum(psi_weight + chi_weight))
    psi_fourier = np.fft.fftn(dpsi, axes=(0, 1, 2))
    chi_fourier = np.fft.fftn(dchi, axes=(0, 1, 2))
    fourier_power = (np.sum(abs(psi_fourier)**2, axis=-1)
                     + np.sum(abs(chi_fourier)**2, axis=-1))
    fourier_order = np.argsort(fourier_power.reshape(-1))[::-1]
    wave_numbers = 2 * np.pi * np.fft.fftfreq(side)

    def fourier_entry(flat_index):
        index = tuple(int(value) for value in np.unravel_index(
            flat_index, fourier_power.shape))
        k = tuple(float(wave_numbers[value]) for value in index)
        symbol = 0.0
        for offset, weight in prior.bond.local.common.base.DIRECTIONS:
            symbol += weight * (1 - np.cos(np.dot(k, offset)))
        return {
            "indices": list(index),
            "wave_numbers": list(k),
            "power_fraction": float(fourier_power[index]
                                    / np.sum(fourier_power)),
            "c19_symbol": float(symbol),
        }
    row.update({
        "spectral_radius": radius,
        "leading_growth_rate_per_unit_time": float(np.log(radius) / dt),
        "linearization_step": jacobian_step,
        "largest_growth_rates": [
            float(np.log(absolute[index]) / dt) for index in order[:12]],
        "leading_mode_corner_overlap_sq": float(np.sum(
            abs(corner.conj().T @ leading)**2).real),
        "leading_mode_radial_overlap_sq": float(np.sum(
            abs(radial.conj().T @ leading)**2).real),
        "leading_mode_phase_overlap_sq": float(np.sum(
            abs(phase.conj().T @ leading)**2).real),
        "leading_mode_psi_fraction": float(np.sum(psi_weight) / total),
        "leading_mode_chi_component_fractions": [
            float(value / total) for value in chi_components],
        "leading_mode_fourier_components": [
            fourier_entry(index) for index in fourier_order[:5]],
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative-local", type=float, default=-0.00425)
    parser.add_argument("--relative-bond", type=float, default=0.007)
    parser.add_argument("--common-gamma", type=float, default=-0.0075)
    parser.add_argument("--psi-berries", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--chi-berries", nargs="+", type=float,
                        default=(-0.01, -0.003, 0.0, 0.003, 0.01))
    parser.add_argument("--psi-density-commons", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-density-commons", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-berry-rho-psis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-berry-rho-chis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-berry-rho-psis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-berry-rho-chis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-density-relatives", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-density-relatives", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--triangle-commons", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--triangle-psis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--triangle-chis", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--parallel-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-gradient-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-gradient-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--area-gradient-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--parallel-potentials", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-gradient-potentials", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-gradient-potentials", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-berry-density-difference-sqs", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--chi-radial-deltas", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--total-density-stiffness-deltas", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--psi-bond-berries", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-bond-berries", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--area-local-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--overlap-common-berries", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--overlap-relative-berries", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--area-relative-gradient-kinetics", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--overlap-bond-common-berries", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--overlap-bond-relative-berries", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--area-plaquette-relative-berries", nargs="+",
                        type=float, default=(0.0,))
    parser.add_argument("--overlap-common-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-orientation-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-orientation-kinetics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--chi-berry-area-weights", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--psi-berry-area-weights", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--normalized-area-commons", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--normalized-area-relatives", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--normalized-area-bond-commons", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--normalized-area-bond-relatives", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--normalized-area-waves", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--directional-area-potentials", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--chi-inertia", type=float, default=229.0)
    parser.add_argument("--jacobian-step", type=float, default=2e-6)
    parser.add_argument("--guess-amplitude-sq", type=float)
    parser.add_argument("--guess-omega-psi", type=float)
    parser.add_argument("--guess-omega-chi", type=float)
    parser.add_argument("--output", default="field_berry_response.json")
    args = parser.parse_args()
    set_chi_inertia(args.chi_inertia)
    dt = 1 / args.rate
    psi, _, qp, qc, _ = prior.bond.local.common.base.rest_state(
        args.side, dt, 1.0, -0.8, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    guess = np.asarray((np.log(prior.bond.local.common.base.norm2(psi).max()),
                        np.angle(qp), np.angle(qc)))
    if args.guess_amplitude_sq is not None:
        guess[0] = np.log(args.guess_amplitude_sq)
    if args.guess_omega_psi is not None:
        guess[1] = args.guess_omega_psi * dt
    if args.guess_omega_chi is not None:
        guess[2] = args.guess_omega_chi * dt
    pools = (
        args.psi_berries, args.chi_berries,
        args.psi_density_commons, args.chi_density_commons,
        args.psi_berry_rho_psis, args.psi_berry_rho_chis,
        args.chi_berry_rho_psis, args.chi_berry_rho_chis,
        args.psi_density_relatives, args.chi_density_relatives,
        args.triangle_commons, args.triangle_psis, args.triangle_chis,
        args.parallel_kinetics,
        args.psi_gradient_kinetics, args.chi_gradient_kinetics,
        args.area_gradient_kinetics, args.parallel_potentials,
        args.psi_gradient_potentials, args.chi_gradient_potentials,
        args.chi_berry_density_difference_sqs, args.chi_radial_deltas,
        args.total_density_stiffness_deltas,
        args.psi_bond_berries, args.chi_bond_berries,
        args.area_local_kinetics, args.overlap_common_berries,
        args.overlap_relative_berries,
        args.area_relative_gradient_kinetics,
        args.overlap_bond_common_berries,
        args.overlap_bond_relative_berries,
        args.area_plaquette_relative_berries,
        args.overlap_common_kinetics,
        args.psi_orientation_kinetics,
        args.chi_orientation_kinetics,
        args.chi_berry_area_weights,
        args.psi_berry_area_weights,
        args.normalized_area_commons,
        args.normalized_area_relatives,
        args.normalized_area_bond_commons,
        args.normalized_area_bond_relatives,
        args.normalized_area_waves,
        args.directional_area_potentials)
    rows = []
    for varied in itertools.product(*pools):
        coefficients = (args.relative_local, args.relative_bond,
                        args.common_gamma, *varied)
        rows.append(classify(args.side, dt, coefficients, guess.copy(),
                             args.jacobian_step))
    result = {
        "status": "FIELD_BERRY_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
