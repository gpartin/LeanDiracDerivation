"""Calculus, independence, and time-reversal checks for gyroscopic actions."""

from __future__ import annotations

import numpy as np

import bond_gyroscopic_probe as bond
import common_gyroscopic_probe as common
import field_berry_probe as field
import gyroscopic_probe as local


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def real_pair(a, b) -> float:
    return float(np.real(np.vdot(a, b)))


def variation_check(name, action, derivatives, fields, directions) -> None:
    epsilon = 2e-7
    plus = [x + epsilon * dx for x, dx in zip(fields, directions)]
    minus = [x - epsilon * dx for x, dx in zip(fields, directions)]
    finite = (action(*plus) - action(*minus)) / (2 * epsilon)
    ppsi, pchi, qpsi, qchi = derivatives(*fields)
    predicted = (real_pair(qpsi, directions[0])
                 + real_pair(qchi, directions[1])
                 + real_pair(ppsi, directions[2])
                 + real_pair(pchi, directions[3]))
    require(abs(finite - predicted) < 3e-5,
            f"{name} first variation mismatch: {finite} vs {predicted}")


def main() -> None:
    rng = np.random.default_rng(404)
    shape = (3, 3, 3, 3)
    fields = [rng.normal(size=shape) + 1j * rng.normal(size=shape)
              for _ in range(4)]
    directions = [rng.normal(size=shape) + 1j * rng.normal(size=shape)
                  for _ in range(4)]
    psi, chi, vp, vc = fields

    def local_action(psi, chi, vp, vc):
        w = np.cross(chi, psi)
        r = np.cross(chi, vp) - np.cross(vc, psi)
        return float(np.vdot(w, r).imag / local.R**2)

    def local_derivatives(psi, chi, vp, vc):
        values = local.gyro_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, 1.0)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check("local relative gyro", local_action, local_derivatives,
                    fields, directions)

    def bond_action(psi, chi, vp, vc):
        w = np.cross(chi, psi)
        r = np.cross(chi, vp) - np.cross(vc, psi)
        return float(np.vdot(w, local.common.base.c19_dirichlet(r)).imag
                     / (local.R**2 * bond.C19_WEIGHT_SUM))

    def bond_derivatives(psi, chi, vp, vc):
        return bond.bond_gyro_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, 1.0)

    variation_check("C19 bond gyro", bond_action, bond_derivatives,
                    fields, directions)

    def common_action(psi, chi, vp, vc):
        w = np.cross(chi, psi)
        wdot = np.cross(chi, vp) + np.cross(vc, psi)
        return float(np.vdot(w, wdot).imag / local.R**2)

    def common_derivatives(psi, chi, vp, vc):
        values = common.common_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, 1.0)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check("common gyro", common_action, common_derivatives,
                    fields, directions)

    def berry_action(psi, chi, vp, vc):
        return float(np.vdot(psi, vp).imag + np.vdot(chi, vc).imag)

    def berry_derivatives(psi, chi, vp, vc):
        return 1j * psi, 1j * chi, -1j * vp, -1j * vc

    variation_check("field Berry", berry_action, berry_derivatives,
                    fields, directions)

    def density_common_action(psi, chi, vp, vc, psi_weight, chi_weight):
        rho_psi = local.common.base.norm2(psi)
        rho_chi = local.common.base.norm2(chi)
        weight = (psi_weight * rho_psi + chi_weight * rho_chi) / local.R**2
        w = np.cross(chi, psi)
        wdot = np.cross(chi, vp) + np.cross(vc, psi)
        common_density = np.imag(np.sum(w.conj() * wdot, axis=-1)) / local.R**2
        return float(np.sum(weight * common_density))

    def density_common_derivatives(psi, chi, vp, vc,
                                   psi_weight, chi_weight):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, psi_weight, chi_weight)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    for name, psi_weight, chi_weight in (
            ("Psi-density common gyro", 0.37, 0.0),
            ("chi-density common gyro", 0.0, -0.23),
            ("mixed-density common gyro", 0.37, -0.23)):
        variation_check(
            name,
            lambda psi, chi, vp, vc, a=psi_weight, b=chi_weight:
                density_common_action(psi, chi, vp, vc, a, b),
            lambda psi, chi, vp, vc, a=psi_weight, b=chi_weight:
                density_common_derivatives(psi, chi, vp, vc, a, b),
            fields, directions)

    def density_field_berry_action(psi, chi, vp, vc,
                                   pp, pc, cp, cc):
        rho_psi = local.common.base.norm2(psi)
        rho_chi = local.common.base.norm2(chi)
        psi_weight = (pp * rho_psi + pc * rho_chi) / local.R**2
        chi_weight = (cp * rho_psi + cc * rho_chi) / local.R**2
        psi_berry_density = np.imag(np.sum(psi.conj() * vp, axis=-1))
        chi_berry_density = np.imag(np.sum(chi.conj() * vc, axis=-1))
        return float(np.sum(psi_weight * psi_berry_density
                            + chi_weight * chi_berry_density))

    def density_field_berry_derivatives(psi, chi, vp, vc,
                                        pp, pc, cp, cc):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pp, pc, cp, cc)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    density_coefficients = (0.19, -0.11, 0.07, -0.13)
    variation_check(
        "density-weighted field Berry",
        lambda psi, chi, vp, vc:
            density_field_berry_action(
                psi, chi, vp, vc, *density_coefficients),
        lambda psi, chi, vp, vc:
            density_field_berry_derivatives(
                psi, chi, vp, vc, *density_coefficients),
        fields, directions)

    def density_relative_action(psi, chi, vp, vc, psi_weight, chi_weight):
        rho_psi = local.common.base.norm2(psi)
        rho_chi = local.common.base.norm2(chi)
        weight = (psi_weight * rho_psi + chi_weight * rho_chi) / local.R**2
        w = np.cross(chi, psi)
        relative_velocity = np.cross(chi, vp) - np.cross(vc, psi)
        density = np.imag(np.sum(
            w.conj() * relative_velocity, axis=-1)) / local.R**2
        return float(np.sum(weight * density))

    def density_relative_derivatives(psi, chi, vp, vc,
                                     psi_weight, chi_weight):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, psi_weight, chi_weight)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "density-weighted relative gyro",
        lambda psi, chi, vp, vc:
            density_relative_action(psi, chi, vp, vc, 0.31, -0.17),
        lambda psi, chi, vp, vc:
            density_relative_derivatives(psi, chi, vp, vc, 0.31, -0.17),
        fields, directions)

    def triangle_common_action(psi, chi, vp, vc):
        area = np.cross(chi, psi)
        area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        common_density = np.imag(np.sum(
            area.conj() * area_velocity, axis=-1)) / local.R**2
        chirality = field.triangle_density(area)
        return float(np.sum(chirality**2 * common_density))

    def triangle_common_derivatives(psi, chi, vp, vc):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check("C19 triangle common gyro", triangle_common_action,
                    triangle_common_derivatives, fields, directions)

    transformed_triangle = triangle_common_action(
        psi.conj(), chi.conj(), -vp.conj(), -vc.conj())
    require(abs(transformed_triangle
                - triangle_common_action(psi, chi, vp, vc)) < 1e-10,
            "triangle common gyro fails antiunitary time reversal")

    def direct_triangle_action(psi, chi, vp, vc,
                               psi_coefficient, chi_coefficient):
        psi_density = field.triangle_density(psi, local.R**2)
        chi_density = field.triangle_density(chi, local.R**2)
        psi_berry = np.imag(np.sum(psi.conj() * vp, axis=-1)) / local.R**2
        chi_berry = np.imag(np.sum(chi.conj() * vc, axis=-1)) / local.R**2
        return float(np.sum(psi_coefficient * psi_density * psi_berry
                            + chi_coefficient * chi_density * chi_berry))

    def direct_triangle_derivatives(psi, chi, vp, vc,
                                    psi_coefficient, chi_coefficient):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            psi_coefficient, chi_coefficient)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    direct_coefficients = (0.23, -0.31)
    variation_check(
        "direct C19 field triangle gyro",
        lambda psi, chi, vp, vc:
            direct_triangle_action(
                psi, chi, vp, vc, *direct_coefficients),
        lambda psi, chi, vp, vc:
            direct_triangle_derivatives(
                psi, chi, vp, vc, *direct_coefficients),
        fields, directions)

    def parallel_kinetic_action(psi, chi, vp, vc):
        relative = (np.sum(chi.conj() * vp, axis=-1)
                    - np.sum(vc.conj() * psi, axis=-1))
        return float(np.sum(abs(relative)**2) / (2 * local.R**2))

    def parallel_kinetic_derivatives(psi, chi, vp, vc):
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check("parallel relative kinetic", parallel_kinetic_action,
                    parallel_kinetic_derivatives, fields, directions)

    def gradient_kinetic_action(psi, chi, vp, vc, psi_eta, chi_eta):
        dpsi = local.common.base.c19_dirichlet(vp)
        dchi = local.common.base.c19_dirichlet(vc)
        return float(
            psi_eta * np.vdot(vp, dpsi).real
            / (2 * bond.C19_WEIGHT_SUM)
            + local.B * chi_eta * np.vdot(vc, dchi).real
            / (2 * bond.C19_WEIGHT_SUM))

    def gradient_kinetic_derivatives(psi, chi, vp, vc,
                                     psi_eta, chi_eta):
        coefficients = (0.0,) * 17 + (psi_eta, chi_eta)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    gradient_coefficients = (0.27, 0.13)
    variation_check(
        "C19 gradient kinetic",
        lambda psi, chi, vp, vc:
            gradient_kinetic_action(
                psi, chi, vp, vc, *gradient_coefficients),
        lambda psi, chi, vp, vc:
            gradient_kinetic_derivatives(
                psi, chi, vp, vc, *gradient_coefficients),
        fields, directions)

    def gradient_potential(psi):
        first = local.common.base.c19_dirichlet(psi)
        return float(np.vdot(first, first).real
                     / (2 * bond.C19_WEIGHT_SUM))

    epsilon = 2e-7
    finite_gradient_potential = (
        gradient_potential(psi + epsilon * directions[0])
        - gradient_potential(psi - epsilon * directions[0])) / (2 * epsilon)
    first = local.common.base.c19_dirichlet(psi)
    expected_gradient = local.common.base.c19_dirichlet(first)
    predicted_gradient_potential = real_pair(
        expected_gradient / bond.C19_WEIGHT_SUM, directions[0])
    require(abs(finite_gradient_potential - predicted_gradient_potential)
            < 3e-5,
            "C19 biharmonic potential first variation mismatch")

    def area_gradient_kinetic_action(psi, chi, vp, vc):
        area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        darea = local.common.base.c19_dirichlet(area_velocity)
        return float(np.vdot(area_velocity, darea).real
                     / (2 * local.R**2 * bond.C19_WEIGHT_SUM))

    def area_gradient_kinetic_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 19 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "C19 area-gradient kinetic", area_gradient_kinetic_action,
        area_gradient_kinetic_derivatives, fields, directions)

    def parallel_potential(psi, chi):
        overlap = np.sum(chi.conj() * psi, axis=-1)
        return float(np.sum(abs(overlap)**2) / 2)

    finite_parallel = (
        parallel_potential(
            psi + epsilon * directions[0], chi + epsilon * directions[1])
        - parallel_potential(
            psi - epsilon * directions[0], chi - epsilon * directions[1])
        ) / (2 * epsilon)
    overlap = np.sum(chi.conj() * psi, axis=-1)[..., None]
    grad_psi = overlap * chi
    grad_chi = overlap.conj() * psi
    predicted_parallel = (real_pair(grad_psi, directions[0])
                          + real_pair(grad_chi, directions[1]))
    require(abs(finite_parallel - predicted_parallel) < 3e-5,
            "parallel-overlap potential first variation mismatch")

    def c19_bond_potential(value):
        derivative = local.common.base.c19_dirichlet(value)
        return float(np.vdot(value, derivative).real
                     / (2 * bond.C19_WEIGHT_SUM))

    finite_bond = (
        c19_bond_potential(chi + epsilon * directions[1])
        - c19_bond_potential(chi - epsilon * directions[1])) / (2 * epsilon)
    predicted_bond = real_pair(
        local.common.base.c19_dirichlet(chi) / bond.C19_WEIGHT_SUM,
        directions[1])
    require(abs(finite_bond - predicted_bond) < 3e-5,
            "C19 direct-bond potential first variation mismatch")

    def density_difference_berry_action(psi, chi, vp, vc):
        difference = (local.common.base.norm2(psi)
                      - local.common.base.norm2(chi))
        berry = np.imag(np.sum(chi.conj() * vc, axis=-1))
        return float(np.sum(difference**2 * berry) / local.R**4)

    def density_difference_berry_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 20 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "quadratic density-difference chi Berry",
        density_difference_berry_action,
        density_difference_berry_derivatives, fields, directions)

    def chi_radial_potential(chi):
        density = local.common.base.norm2(chi)
        return float(np.sum((density - local.R**2)**2) / 4)

    radial_epsilon = 2e-6
    finite_chi_radial = (
        chi_radial_potential(chi + radial_epsilon * directions[1])
        - chi_radial_potential(chi - radial_epsilon * directions[1])
        ) / (2 * radial_epsilon)
    density_chi = local.common.base.norm2(chi)[..., None]
    predicted_chi_radial = real_pair(
        (density_chi - local.R**2) * chi, directions[1])
    require(abs(finite_chi_radial - predicted_chi_radial) < 3e-5,
            "chi radial-potential first variation mismatch")

    def total_density_potential(psi, chi):
        mismatch = (local.common.base.norm2(psi)
                    + local.common.base.norm2(chi) - local.R**2)
        return float(np.sum(mismatch**2) / 4)

    finite_total_density = (
        total_density_potential(
            psi + radial_epsilon * directions[0],
            chi + radial_epsilon * directions[1])
        - total_density_potential(
            psi - radial_epsilon * directions[0],
            chi - radial_epsilon * directions[1])) / (2 * radial_epsilon)
    mismatch = (local.common.base.norm2(psi)
                + local.common.base.norm2(chi) - local.R**2)[..., None]
    predicted_total_density = (
        real_pair(mismatch * psi, directions[0])
        + real_pair(mismatch * chi, directions[1]))
    require(abs(finite_total_density - predicted_total_density) < 3e-5,
            "combined-density potential first variation mismatch")

    def direct_bond_berry_action(psi, chi, vp, vc,
                                 psi_coefficient, chi_coefficient):
        return float((
            psi_coefficient * np.vdot(
                psi, local.common.base.c19_dirichlet(vp)).imag
            + chi_coefficient * np.vdot(
                chi, local.common.base.c19_dirichlet(vc)).imag
        ) / bond.C19_WEIGHT_SUM)

    def direct_bond_berry_derivatives(psi, chi, vp, vc,
                                      psi_coefficient, chi_coefficient):
        coefficients = ((0.0,) * 21
                        + (psi_coefficient, chi_coefficient))
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    bond_berry_coefficients = (0.17, -0.29)
    variation_check(
        "direct C19 field Berry",
        lambda psi, chi, vp, vc:
            direct_bond_berry_action(
                psi, chi, vp, vc, *bond_berry_coefficients),
        lambda psi, chi, vp, vc:
            direct_bond_berry_derivatives(
                psi, chi, vp, vc, *bond_berry_coefficients),
        fields, directions)

    def area_local_kinetic_action(psi, chi, vp, vc):
        area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        return float(np.vdot(area_velocity, area_velocity).real
                     / (2 * local.R**2))

    def area_local_kinetic_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 23 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check("local area kinetic", area_local_kinetic_action,
                    area_local_kinetic_derivatives, fields, directions)

    def overlap_berry_action(psi, chi, vp, vc, relative):
        overlap = np.sum(chi.conj() * psi, axis=-1)
        if relative:
            overlap_velocity = (
                np.sum(chi.conj() * vp, axis=-1)
                - np.sum(vc.conj() * psi, axis=-1))
        else:
            overlap_velocity = (
                np.sum(chi.conj() * vp, axis=-1)
                + np.sum(vc.conj() * psi, axis=-1))
        return float(np.sum(overlap.conj() * overlap_velocity).imag)

    def overlap_berry_derivatives(psi, chi, vp, vc, relative):
        coefficients = ((0.0,) * 24
                        + ((0.0, 1.0) if relative else (1.0, 0.0)))
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    for name, relative in (("common overlap Berry", False),
                           ("relative overlap Berry", True)):
        variation_check(
            name,
            lambda psi, chi, vp, vc, relative=relative:
                overlap_berry_action(psi, chi, vp, vc, relative),
            lambda psi, chi, vp, vc, relative=relative:
                overlap_berry_derivatives(psi, chi, vp, vc, relative),
            fields, directions)

    def relative_area_gradient_kinetic_action(psi, chi, vp, vc):
        relative_area_velocity = (np.cross(chi, vp)
                                  - np.cross(vc, psi))
        filtered = local.common.base.c19_dirichlet(
            relative_area_velocity)
        return float(np.vdot(relative_area_velocity, filtered).real
                     / (2 * local.R**2 * bond.C19_WEIGHT_SUM))

    def relative_area_gradient_kinetic_derivatives(psi, chi, vp, vc):
        # all_derivatives omits the five position-only coefficients carried by
        # del_residual, so this is its 27th action-derivative coefficient.
        coefficients = (0.0,) * 26 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "C19 relative-area-gradient kinetic",
        relative_area_gradient_kinetic_action,
        relative_area_gradient_kinetic_derivatives,
        fields, directions)

    def overlap_bond_common_berry_action(psi, chi, vp, vc):
        overlap = np.sum(chi.conj() * psi, axis=-1)
        overlap_velocity = (np.sum(chi.conj() * vp, axis=-1)
                            + np.sum(vc.conj() * psi, axis=-1))
        filtered_velocity = local.common.base.c19_dirichlet(
            overlap_velocity)
        return float(np.vdot(overlap, filtered_velocity).imag
                     / bond.C19_WEIGHT_SUM)

    def overlap_bond_common_berry_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 27 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "C19 common-overlap bond Berry",
        overlap_bond_common_berry_action,
        overlap_bond_common_berry_derivatives,
        fields, directions)

    def overlap_bond_relative_berry_action(psi, chi, vp, vc):
        overlap = np.sum(chi.conj() * psi, axis=-1)
        relative_overlap = (np.sum(chi.conj() * vp, axis=-1)
                            - np.sum(vc.conj() * psi, axis=-1))
        filtered_velocity = local.common.base.c19_dirichlet(
            relative_overlap)
        return float(np.vdot(overlap, filtered_velocity).imag
                     / bond.C19_WEIGHT_SUM)

    def overlap_bond_relative_berry_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 28 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "C19 relative-overlap bond Berry",
        overlap_bond_relative_berry_action,
        overlap_bond_relative_berry_derivatives,
        fields, directions)

    def area_plaquette_relative_berry_action(psi, chi, vp, vc):
        area = np.cross(chi, psi)
        relative = np.cross(chi, vp) - np.cross(vc, psi)
        filtered_relative = local.common.base.plaquette_dirichlet(relative)
        return float(np.vdot(area, filtered_relative).imag
                     / (local.R**2
                        * len(local.common.base.PLAQUETTES)))

    def area_plaquette_relative_berry_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 29 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "plaquette relative-area Berry",
        area_plaquette_relative_berry_action,
        area_plaquette_relative_berry_derivatives,
        fields, directions)

    def overlap_common_kinetic_action(psi, chi, vp, vc):
        overlap_velocity = (np.sum(chi.conj() * vp, axis=-1)
                            + np.sum(vc.conj() * psi, axis=-1))
        return float(np.sum(abs(overlap_velocity)**2)
                     / (2 * local.R**2))

    def overlap_common_kinetic_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 30 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "common-overlap kinetic",
        overlap_common_kinetic_action,
        overlap_common_kinetic_derivatives,
        fields, directions)

    def psi_orientation_kinetic_action(psi, chi, vp, vc):
        oriented_velocity = np.cross(chi, vp)
        return float(np.vdot(oriented_velocity, oriented_velocity).real
                     / (2 * local.R**2))

    def psi_orientation_kinetic_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 31 + (1.0, 0.0)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "Psi orientation kinetic",
        psi_orientation_kinetic_action,
        psi_orientation_kinetic_derivatives,
        fields, directions)

    def chi_orientation_kinetic_action(psi, chi, vp, vc):
        oriented_velocity = np.cross(vc, psi)
        return float(np.vdot(oriented_velocity, oriented_velocity).real
                     / (2 * local.R**2))

    def chi_orientation_kinetic_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 31 + (0.0, 1.0)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "chi orientation kinetic",
        chi_orientation_kinetic_action,
        chi_orientation_kinetic_derivatives,
        fields, directions)

    def chi_berry_area_action(psi, chi, vp, vc):
        area_density = local.common.base.norm2(np.cross(chi, psi))
        berry_density = np.imag(np.sum(chi.conj() * vc, axis=-1))
        return float(np.sum(area_density * berry_density) / local.R**4)

    def chi_berry_area_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 33 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "oriented-area-weighted chi Berry",
        chi_berry_area_action,
        chi_berry_area_derivatives,
        fields, directions)

    def psi_berry_area_action(psi, chi, vp, vc):
        area_density = local.common.base.norm2(np.cross(chi, psi))
        berry_density = np.imag(np.sum(psi.conj() * vp, axis=-1))
        return float(np.sum(area_density * berry_density) / local.R**4)

    def psi_berry_area_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 34 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "oriented-area-weighted Psi Berry",
        psi_berry_area_action,
        psi_berry_area_derivatives,
        fields, directions)

    def normalized_area_action(psi, chi, vp, vc, relative):
        area = np.cross(chi, psi)
        if relative:
            area_velocity = np.cross(chi, vp) - np.cross(vc, psi)
        else:
            area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        density = local.common.base.norm2(area)
        berry = np.imag(np.sum(area.conj() * area_velocity, axis=-1))
        return float(np.sum(local.R**2 * berry
                            / (local.R**4 + density)))

    def normalized_area_derivatives(psi, chi, vp, vc, relative):
        coefficients = ((0.0,) * 35
                        + ((0.0, 1.0) if relative else (1.0, 0.0)))
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    for name, relative in (("normalized common-area Berry", False),
                           ("normalized relative-area Berry", True)):
        variation_check(
            name,
            lambda psi, chi, vp, vc, relative=relative:
                normalized_area_action(psi, chi, vp, vc, relative),
            lambda psi, chi, vp, vc, relative=relative:
                normalized_area_derivatives(psi, chi, vp, vc, relative),
            fields, directions)

    def normalized_bond_area_action(psi, chi, vp, vc, relative):
        area = np.cross(chi, psi)
        if relative:
            area_velocity = np.cross(chi, vp) - np.cross(vc, psi)
        else:
            area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        density = local.common.base.norm2(area)[..., None]
        denominator = local.R**4 + density
        h = local.R / np.sqrt(denominator)
        h_prime = -local.R / (2 * denominator**1.5)
        radial_velocity = np.real(np.sum(
            area.conj() * area_velocity, axis=-1))[..., None]
        normalized_area = h * area
        normalized_velocity = (h * area_velocity
                               + 2 * h_prime * radial_velocity * area)
        return float(np.vdot(
            normalized_area,
            local.common.base.c19_dirichlet(normalized_velocity)).imag
            / bond.C19_WEIGHT_SUM)

    def normalized_bond_area_derivatives(psi, chi, vp, vc, relative):
        coefficients = ((0.0,) * 37
                        + ((0.0, 1.0) if relative else (1.0, 0.0)))
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    for name, relative in (
            ("normalized C19 common-area Berry", False),
            ("normalized C19 relative-area Berry", True)):
        variation_check(
            name,
            lambda psi, chi, vp, vc, relative=relative:
                normalized_bond_area_action(psi, chi, vp, vc, relative),
            lambda psi, chi, vp, vc, relative=relative:
                normalized_bond_area_derivatives(
                    psi, chi, vp, vc, relative),
            fields, directions)

    def normalized_area_wave_action(psi, chi, vp, vc):
        area = np.cross(chi, psi)
        area_velocity = np.cross(chi, vp) + np.cross(vc, psi)
        density = local.common.base.norm2(area)[..., None]
        denominator = local.R**4 + density
        h = local.R / np.sqrt(denominator)
        h_prime = -local.R / (2 * denominator**1.5)
        radial_velocity = np.real(np.sum(
            area.conj() * area_velocity, axis=-1))[..., None]
        normalized_area = h * area
        normalized_velocity = (h * area_velocity
                               + 2 * h_prime * radial_velocity * area)
        d_velocity = local.common.base.c19_dirichlet(normalized_velocity)
        d_area = local.common.base.c19_dirichlet(normalized_area)
        d2_area = local.common.base.c19_dirichlet(d_area)
        kinetic = np.vdot(normalized_velocity, d_velocity).real
        potential = np.vdot(normalized_area, d2_area).real
        return float((kinetic - potential) / (2 * bond.C19_WEIGHT_SUM))

    def normalized_area_wave_derivatives(psi, chi, vp, vc):
        coefficients = (0.0,) * 39 + (1.0,)
        values = field.all_derivatives(
            psi - vp / 2, chi - vc / 2,
            psi + vp / 2, chi + vc / 2, 1.0, *coefficients)
        return (values[0] - vp, values[1] - local.B * vc,
                values[2], values[3])

    variation_check(
        "normalized C19 area wave",
        normalized_area_wave_action,
        normalized_area_wave_derivatives,
        fields, directions)

    def directional_area_energy(psi, chi):
        area = np.cross(chi, psi)
        total = 0.0
        for offset, weight in local.common.base.DIRECTIONS:
            direction = np.asarray(offset, dtype=float)
            direction /= np.linalg.norm(direction)
            difference = area - local.common.base.shift(area, offset)
            longitudinal = np.sum(difference * direction, axis=-1)
            total += weight * np.sum(abs(longitudinal)**2)
        return float(total / (2 * local.R**2))

    epsilon = 2e-7
    energy_plus = directional_area_energy(
        psi + epsilon * directions[0], chi + epsilon * directions[1])
    energy_minus = directional_area_energy(
        psi - epsilon * directions[0], chi - epsilon * directions[1])
    finite_energy = (energy_plus - energy_minus) / (2 * epsilon)
    gradient = field.directional_area_gradient(np.cross(chi, psi), 1.0)
    force_psi = -np.cross(gradient, chi.conj())
    force_chi = -np.cross(psi.conj(), gradient)
    predicted_energy = -(real_pair(force_psi, directions[0])
                         + real_pair(force_chi, directions[1]))
    require(abs(finite_energy - predicted_energy) < 3e-5,
            "directional area potential variation mismatch: "
            f"{finite_energy} vs {predicted_energy}")

    def normalized_cross_covariance_energy(psi, chi):
        relation = chi[..., :, None] * psi.conj()[..., None, :]
        density = np.sum(abs(relation)**2, axis=(-2, -1), keepdims=True)
        normalized = (local.R * relation
                      / np.sqrt(local.R**4 + density))
        return float(np.vdot(
            normalized,
            local.common.base.c19_dirichlet(normalized)).real
            / (2 * bond.C19_WEIGHT_SUM))

    energy_plus = normalized_cross_covariance_energy(
        psi + epsilon * directions[0], chi + epsilon * directions[1])
    energy_minus = normalized_cross_covariance_energy(
        psi - epsilon * directions[0], chi - epsilon * directions[1])
    finite_energy = (energy_plus - energy_minus) / (2 * epsilon)
    gradient_psi, gradient_chi = (
        field.normalized_cross_covariance_bond_gradients(psi, chi, 1.0))
    predicted_energy = (real_pair(gradient_psi, directions[0])
                        + real_pair(gradient_chi, directions[1]))
    require(abs(finite_energy - predicted_energy) < 3e-5,
            "normalized cross-covariance bond variation mismatch: "
            f"{finite_energy} vs {predicted_energy}")

    # A total-derivative one-form is closed.  This explicit nonzero exterior
    # derivative proves the local relative gyro is not merely a boundary term.
    qpsi, qchi = fields[:2]
    upsi, uchi, vpsi, vchi = directions

    def one_form(psi, chi, dpsi, dchi):
        w = np.cross(chi, psi)
        r = np.cross(chi, dpsi) - np.cross(dchi, psi)
        return float(np.vdot(w, r).imag / local.R**2)

    epsilon = 2e-7
    du_av = (one_form(qpsi + epsilon * upsi, qchi + epsilon * uchi,
                      vpsi, vchi)
             - one_form(qpsi - epsilon * upsi, qchi - epsilon * uchi,
                        vpsi, vchi)) / (2 * epsilon)
    dv_au = (one_form(qpsi + epsilon * vpsi, qchi + epsilon * vchi,
                      upsi, uchi)
             - one_form(qpsi - epsilon * vpsi, qchi - epsilon * vchi,
                        upsi, uchi)) / (2 * epsilon)
    require(abs(du_av - dv_au) > 1e-3,
            "local gyroscopic one-form unexpectedly appears closed")

    # Antiunitary time reversal: q -> q*, qdot -> -qdot* leaves each
    # imaginary velocity-linear invariant unchanged.
    require(abs(local_action(psi.conj(), chi.conj(), -vp.conj(), -vc.conj())
                - local_action(psi, chi, vp, vc)) < 1e-10,
            "local gyro fails antiunitary time reversal")
    require(abs(common_action(psi.conj(), chi.conj(), -vp.conj(), -vc.conj())
                - common_action(psi, chi, vp, vc)) < 1e-10,
            "common gyro fails antiunitary time reversal")

    print("V4_GYROSCOPIC_ACTION_CALCULUS_PASS")


if __name__ == "__main__":
    main()
