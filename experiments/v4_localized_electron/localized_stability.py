"""Full native tangent stability of the exact V4 localized rest orbit.

This is a numerical classification of the unchanged action, not a Lean proof.
It retains every Psi/chi component, every site, both committed layers, and the
exact internal co-rotating frame used by the symbolic rest-orbit theorem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
R = 19.0
B = 1197.0
AXES = (0, 1, 2)
DIRECTIONS = [
    (offset, 3 - sum(abs(x) for x in offset))
    for offset in itertools.product((-1, 0, 1), repeat=3)
    if 1 <= sum(abs(x) for x in offset) <= 2
]
PLAQUETTES = [
    (tuple(sign_a if coordinate == axis_a else 0 for coordinate in AXES),
     tuple(sign_b if coordinate == axis_b else 0 for coordinate in AXES))
    for axis_a, axis_b in itertools.combinations(AXES, 2)
    for sign_a in (-1, 1) for sign_b in (-1, 1)
]


def shift(field: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    return np.roll(field, tuple(-x for x in offset), axis=AXES)


def norm2(field: np.ndarray) -> np.ndarray:
    return np.sum(abs(field) ** 2, axis=-1)


def bond_gradient(area: np.ndarray) -> np.ndarray:
    laplacian = sum(weight * (area - shift(area, offset))
                    for offset, weight in DIRECTIONS)
    return (2.0 / R**2) * laplacian


def nonlinear_bond_gradient(area: np.ndarray, quartic: float,
                            sextic: float = 0.0) -> np.ndarray:
    result = np.zeros_like(area)
    for offset, weight in DIRECTIONS:
        difference = area - shift(area, offset)
        size = np.sum(abs(difference)**2, axis=-1)[..., None]
        derivative = (1 + 2 * quartic * size / R**4
                      + 3 * sextic * size**2 / R**8)
        result += weight * derivative * difference
    return (2 / R**2) * result


def c19_dirichlet(field: np.ndarray) -> np.ndarray:
    return sum(weight * (field - shift(field, offset))
               for offset, weight in DIRECTIONS)


def plaquette_dirichlet(field: np.ndarray) -> np.ndarray:
    """Sum P^dagger P over all 12 signed elementary cube plaquettes."""
    result = np.zeros_like(field)
    for a, b in PLAQUETTES:
        ab = tuple(x + y for x, y in zip(a, b))
        curvature = shift(field, ab) - shift(field, a) - shift(field, b) + field
        minus_a = tuple(-x for x in a)
        minus_b = tuple(-x for x in b)
        minus_ab = tuple(-x for x in ab)
        result += (shift(curvature, minus_ab) - shift(curvature, minus_a)
                   - shift(curvature, minus_b) + curvature)
    return result


def repair_force(psi: np.ndarray, chi: np.ndarray, strength: float):
    """Force from U8 = strength/(4 R^4) rhoPsi^2 (rhoChi-rhoPsi)^2."""
    p, c = norm2(psi)[..., None], norm2(chi)[..., None]
    difference = c - p
    psi_coefficient = p * difference * (c - 2 * p)
    chi_coefficient = p**2 * difference
    return (-strength * psi_coefficient * psi / R**4,
            -strength * chi_coefficient * chi / (B * R**4))


def parallel_force(psi: np.ndarray, chi: np.ndarray, strength: float):
    """Force from the generic positive local energy strength |chi^dag psi|^2."""
    overlap = np.sum(chi.conj() * psi, axis=-1)[..., None]
    return (-strength * chi * overlap,
            -strength * psi * overlap.conj() / B)


def onsite_area_polynomial_force(psi: np.ndarray, chi: np.ndarray,
                                 quartic: float, sextic: float):
    """Force from a bounded-capable polynomial in |chi wedge psi|^2."""
    area = np.cross(chi, psi)
    size = norm2(area)[..., None]
    gradient = (quartic * size / R**4
                + sextic * size**2 / R**8) * area
    return (-np.cross(gradient, chi.conj()),
            -np.cross(psi.conj(), gradient) / B)


def c19_curvature_force(psi: np.ndarray, chi: np.ndarray, strength: float):
    """Force from strength ||D19 (chi wedge psi)||^2/(2 R^4)."""
    area = np.cross(chi, psi)
    gradient = strength * (2 / R**4) * c19_dirichlet(
        c19_dirichlet(area))
    return (-np.cross(gradient, chi.conj()),
            -np.cross(psi.conj(), gradient) / B)


def plaquette_curvature_force(psi: np.ndarray, chi: np.ndarray,
                              strength: float):
    """Force from a positive, cubic sum of shared-corner plaquette bends."""
    area = np.cross(chi, psi)
    gradient = strength * (2 / (R**2 * len(PLAQUETTES))) * plaquette_dirichlet(area)
    return (-np.cross(gradient, chi.conj()),
            -np.cross(psi.conj(), gradient) / B)


def force(psi: np.ndarray, chi: np.ndarray, area_scale: float = 1.0,
          repair_strength: float = 0.0,
          bond_quartic: float = 0.0,
          bond_sextic: float = 0.0,
          parallel_strength: float = 0.0,
          local_cross_scale: float = 1.0,
          local_self_scale: float = 1.0,
          onsite_area_quartic: float = 0.0,
          onsite_area_sextic: float = 0.0,
          c19_curvature_strength: float = 0.0,
          plaquette_curvature_strength: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    area_force = nonlinear_bond_gradient(
        np.cross(chi, psi), bond_quartic, bond_sextic)
    p, c = norm2(psi)[..., None], norm2(chi)[..., None]
    local = (-(local_self_scale * p + local_cross_scale * c) * psi,
             -(local_self_scale * (c - R**2)
               + local_cross_scale * p) * chi / B)
    area = (-np.cross(area_force, chi.conj()),
            -np.cross(psi.conj(), area_force) / B)
    repair = repair_force(psi, chi, repair_strength)
    parallel = parallel_force(psi, chi, parallel_strength)
    onsite = onsite_area_polynomial_force(
        psi, chi, onsite_area_quartic, onsite_area_sextic)
    curvature = c19_curvature_force(psi, chi, c19_curvature_strength)
    plaquette = plaquette_curvature_force(
        psi, chi, plaquette_curvature_strength)
    return (local[0] + area_scale * area[0] + repair[0] + parallel[0]
            + onsite[0] + curvature[0] + plaquette[0],
            local[1] + area_scale * area[1] + repair[1] + parallel[1]
            + onsite[1] + curvature[1] + plaquette[1])


def tangent_force(psi: np.ndarray, chi: np.ndarray,
                  dpsi: np.ndarray, dchi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density = (norm2(psi) + norm2(chi))[..., None]
    ddensity = 2 * np.real(np.sum(psi.conj() * dpsi + chi.conj() * dchi,
                                  axis=-1))[..., None]
    area_force = bond_gradient(np.cross(chi, psi))
    darea_force = bond_gradient(np.cross(dchi, psi) + np.cross(chi, dpsi))
    return (
        -(ddensity * psi + density * dpsi
          + np.cross(darea_force, chi.conj())
          + np.cross(area_force, dchi.conj())),
        -(ddensity * chi + (density - R**2) * dchi
          + np.cross(dpsi.conj(), area_force)
          + np.cross(psi.conj(), darea_force)) / B,
    )


def tangent_force_parts(psi: np.ndarray, chi: np.ndarray,
                        dpsi: np.ndarray, dchi: np.ndarray,
                        bond_quartic: float, bond_sextic: float,
                        local_cross_scale: float, local_self_scale: float):
    p, c = norm2(psi)[..., None], norm2(chi)[..., None]
    dp = 2 * np.real(np.sum(psi.conj() * dpsi, axis=-1))[..., None]
    dc = 2 * np.real(np.sum(chi.conj() * dchi, axis=-1))[..., None]
    psi_coefficient = local_self_scale * p + local_cross_scale * c
    chi_coefficient = (local_self_scale * (c - R**2)
                       + local_cross_scale * p)
    dpsi_coefficient = local_self_scale * dp + local_cross_scale * dc
    dchi_coefficient = local_self_scale * dc + local_cross_scale * dp
    area = np.cross(chi, psi)
    darea = np.cross(dchi, psi) + np.cross(chi, dpsi)
    area_force = nonlinear_bond_gradient(area, bond_quartic, bond_sextic)
    darea_force = np.zeros_like(area)
    for offset, weight in DIRECTIONS:
        difference = area - shift(area, offset)
        ddifference = darea - shift(darea, offset)
        size = np.sum(abs(difference)**2, axis=-1)[..., None]
        dsize = 2 * np.real(np.sum(difference.conj() * ddifference,
                                   axis=-1))[..., None]
        first = (1 + 2 * bond_quartic * size / R**4
                 + 3 * bond_sextic * size**2 / R**8)
        second = (2 * bond_quartic / R**4
                  + 6 * bond_sextic * size / R**8)
        darea_force += weight * (first * ddifference
                                 + second * dsize * difference)
    darea_force *= 2 / R**2
    local = (
        -(dpsi_coefficient * psi + psi_coefficient * dpsi),
        -(dchi_coefficient * chi + chi_coefficient * dchi) / B,
    )
    area = (
        -(np.cross(darea_force, chi.conj()) + np.cross(area_force, dchi.conj())),
        -(np.cross(dpsi.conj(), area_force) + np.cross(psi.conj(), darea_force)) / B,
    )
    return local, area


def repair_tangent_force(psi: np.ndarray, chi: np.ndarray,
                         dpsi: np.ndarray, dchi: np.ndarray):
    p, c = norm2(psi)[..., None], norm2(chi)[..., None]
    dp = 2 * np.real(np.sum(psi.conj() * dpsi, axis=-1))[..., None]
    dc = 2 * np.real(np.sum(chi.conj() * dchi, axis=-1))[..., None]
    difference, ddifference = c - p, dc - dp
    a = p * difference * (c - 2 * p)
    da = (dp * difference * (c - 2 * p)
          + p * ddifference * (c - 2 * p)
          + p * difference * (dc - 2 * dp))
    h = p**2 * difference
    dh = 2 * p * dp * difference + p**2 * ddifference
    return (-(da * psi + a * dpsi) / R**4,
            -(dh * chi + h * dchi) / (B * R**4))


def parallel_tangent_force(psi: np.ndarray, chi: np.ndarray,
                           dpsi: np.ndarray, dchi: np.ndarray):
    overlap = np.sum(chi.conj() * psi, axis=-1)[..., None]
    doverlap = np.sum(dchi.conj() * psi + chi.conj() * dpsi,
                      axis=-1)[..., None]
    return (-(dchi * overlap + chi * doverlap),
            -(dpsi * overlap.conj() + psi * doverlap.conj()) / B)


def onsite_area_polynomial_tangent_force(psi: np.ndarray, chi: np.ndarray,
                                         dpsi: np.ndarray, dchi: np.ndarray,
                                         order: int):
    area = np.cross(chi, psi)
    darea = np.cross(dchi, psi) + np.cross(chi, dpsi)
    size = norm2(area)[..., None]
    dsize = 2 * np.real(np.sum(area.conj() * darea, axis=-1))[..., None]
    if order == 2:
        first, second = size / R**4, np.ones_like(size) / R**4
    elif order == 3:
        first, second = size**2 / R**8, 2 * size / R**8
    else:
        raise ValueError("onsite area order must be 2 or 3")
    gradient = first * area
    dgradient = first * darea + second * dsize * area
    return (
        -(np.cross(dgradient, chi.conj())
          + np.cross(gradient, dchi.conj())),
        -(np.cross(dpsi.conj(), gradient)
          + np.cross(psi.conj(), dgradient)) / B,
    )


def c19_curvature_tangent_force(psi: np.ndarray, chi: np.ndarray,
                                dpsi: np.ndarray, dchi: np.ndarray):
    area = np.cross(chi, psi)
    darea = np.cross(dchi, psi) + np.cross(chi, dpsi)
    gradient = (2 / R**4) * c19_dirichlet(c19_dirichlet(area))
    dgradient = (2 / R**4) * c19_dirichlet(c19_dirichlet(darea))
    return (
        -(np.cross(dgradient, chi.conj())
          + np.cross(gradient, dchi.conj())),
        -(np.cross(dpsi.conj(), gradient)
          + np.cross(psi.conj(), dgradient)) / B,
    )


def plaquette_curvature_tangent_force(psi: np.ndarray, chi: np.ndarray,
                                      dpsi: np.ndarray, dchi: np.ndarray):
    area = np.cross(chi, psi)
    darea = np.cross(dchi, psi) + np.cross(chi, dpsi)
    scale = 2 / (R**2 * len(PLAQUETTES))
    gradient = scale * plaquette_dirichlet(area)
    dgradient = scale * plaquette_dirichlet(darea)
    return (
        -(np.cross(dgradient, chi.conj())
          + np.cross(gradient, dchi.conj())),
        -(np.cross(dpsi.conj(), gradient)
          + np.cross(psi.conj(), dgradient)) / B,
    )


def spectral_geometry_tangent_force(psi: np.ndarray, chi: np.ndarray,
                                    dpsi: np.ndarray, dchi: np.ndarray):
    """Hessian of ||D19(D19-27)W||^2/(R^2 27^2) at its zero set."""
    darea = np.cross(dchi, psi) + np.cross(chi, dpsi)
    first = c19_dirichlet(c19_dirichlet(darea) - 27 * darea)
    second = c19_dirichlet(c19_dirichlet(first) - 27 * first)
    gradient = (2 / (R**2 * 27**2)) * second
    return (-np.cross(gradient, chi.conj()),
            -np.cross(psi.conj(), gradient) / B)


def balanced_values(dt: float, local_self_scale: float) -> tuple[float, float, float]:
    gap = local_self_scale * R**2
    psi_frequency_sq = (B * gap + dt**2 * gap**2 / 4) / (
        B - 1 + dt**2 * gap / 2)
    chi_frequency_sq = (psi_frequency_sq - gap) / B
    amplitude_sq = psi_frequency_sq / (1 + local_self_scale + 54 / R**2)
    return amplitude_sq, psi_frequency_sq, chi_frequency_sq


def phase_step(dt: float, frequency_sq: float) -> complex:
    centered_rate_sq = frequency_sq * (1 - dt**2 * frequency_sq / 4)
    return 1 - dt**2 * frequency_sq / 2 + 1j * dt * np.sqrt(centered_rate_sq)


def carrier_fields(side: int, amplitude_sq: float):
    psi = np.zeros((side, side, side, 3), dtype=complex)
    chi = np.zeros_like(psi)
    chi[..., 2] = R
    omega = -0.5 + 0.5j * np.sqrt(3)
    a, b = 1 / np.sqrt(3), np.sqrt(2 / 3)
    rays = [np.asarray(x, dtype=complex) for x in (
        (1, 0, 0), (a, b, 0), (a, b * omega, 0),
        (a, b * omega.conjugate(), 0),
    )]
    rays += [-x for x in reversed(rays)]
    twist = (1, -1j, -1, 1j, -1, 1j, 1, -1j)
    center = side // 2 - 1
    for index, corner in enumerate(itertools.product((0, 1), repeat=3)):
        site = tuple(center + x for x in corner)
        ray = rays[index]
        psi[site] = np.sqrt(amplitude_sq) * twist[index] * ray
        chi[site] = np.sqrt(amplitude_sq) * np.asarray(
            (-ray[1].conjugate(), ray[0].conjugate(), 0), dtype=complex)
    return psi, chi


def nonlinear_area_stiffness(side: int, order: int) -> float:
    psi, chi = carrier_fields(side, 1.0)
    area = np.cross(chi, psi)
    linear = nonlinear_bond_gradient(area, 0.0, 0.0)
    if order == 2:
        full = nonlinear_bond_gradient(area, 1.0, 0.0)
    elif order == 3:
        full = nonlinear_bond_gradient(area, 0.0, 1.0)
    else:
        raise ValueError("bond order must be 2 or 3")
    nonlinear_force = -np.cross(full - linear, chi.conj())
    return float(-np.vdot(psi, nonlinear_force).real
                 / np.vdot(psi, psi).real)


def onsite_area_stiffness(side: int, order: int) -> float:
    psi, chi = carrier_fields(side, 1.0)
    force_psi, _ = onsite_area_polynomial_force(
        psi, chi, float(order == 2), float(order == 3))
    return float(-np.vdot(psi, force_psi).real
                 / np.vdot(psi, psi).real)


def c19_curvature_stiffness(side: int) -> float:
    psi, chi = carrier_fields(side, 1.0)
    force_psi, _ = c19_curvature_force(psi, chi, 1.0)
    return float(-np.vdot(psi, force_psi).real
                 / np.vdot(psi, psi).real)


def plaquette_curvature_stiffness(side: int) -> float:
    psi, chi = carrier_fields(side, 1.0)
    force_psi, _ = plaquette_curvature_force(psi, chi, 1.0)
    return float(-np.vdot(psi, force_psi).real
                 / np.vdot(psi, psi).real)


def positive_roots(function, upper: float) -> list[float]:
    """Return all detected positive roots on a deterministic mixed grid."""
    linear = np.linspace(0.0, upper, 2001)
    geometric = np.geomspace(max(upper * 1e-10, 1e-14), upper, 1001)
    grid = np.unique(np.concatenate((linear, geometric)))
    values = [float(function(x)) for x in grid]
    roots = []
    for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fl == 0 and left > 0:
            roots.append(float(left))
        if fl * fr < 0:
            roots.append(float(brentq(function, left, right)))
    if values[-1] == 0:
        roots.append(float(grid[-1]))
    return sorted({round(root, 13) for root in roots if root > 1e-12})


def rest_state(side: int, dt: float, area_scale: float, bond_quartic: float,
               bond_sextic: float, local_cross_scale: float,
               local_self_scale: float, onsite_area_quartic: float,
               onsite_area_sextic: float, c19_curvature_strength: float,
               plaquette_curvature_strength: float):
    _, psi_frequency_sq, chi_frequency_sq = balanced_values(
        dt, local_self_scale)
    h1 = 54 / R**2
    h4 = nonlinear_area_stiffness(side, 2)
    h6 = nonlinear_area_stiffness(side, 3)
    hlocal4 = onsite_area_stiffness(side, 2)
    hlocal6 = onsite_area_stiffness(side, 3)
    hcurvature = c19_curvature_stiffness(side)
    hplaquette = plaquette_curvature_stiffness(side)
    equation = lambda x: ((local_self_scale + local_cross_scale) * x
        + area_scale * (
        h1 * x + bond_quartic * h4 * x**3 + bond_sextic * h6 * x**5)
        + onsite_area_quartic * hlocal4 * x**3
        + onsite_area_sextic * hlocal6 * x**5
        + c19_curvature_strength * hcurvature * x
        + plaquette_curvature_strength * hplaquette * x
        - psi_frequency_sq)
    roots = positive_roots(equation, max(4 * psi_frequency_sq, R**2 * 4))
    if not roots:
        raise ValueError("no positive exact-rest amplitude for this action")
    baseline = psi_frequency_sq / (
        local_self_scale + local_cross_scale + area_scale * h1)
    amplitude_sq = min(roots, key=lambda root: abs(root - baseline))
    psi, chi = carrier_fields(side, amplitude_sq)
    qp = phase_step(dt, psi_frequency_sq)
    qc = phase_step(dt, chi_frequency_sq)
    return psi, chi, qp, qc, roots


def pack(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    return np.stack((psi.real, psi.imag, np.sqrt(B) * chi.real,
                     np.sqrt(B) * chi.imag), axis=-2).reshape(-1)


def unpack(vector: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    data = vector.reshape(side, side, side, 4, 3)
    return (data[..., 0, :] + 1j * data[..., 1, :],
            (data[..., 2, :] + 1j * data[..., 3, :]) / np.sqrt(B))


def tangent_matrices(psi: np.ndarray, chi: np.ndarray, bond_quartic: float,
                     bond_sextic: float, local_cross_scale: float,
                     local_self_scale: float):
    side = psi.shape[0]
    dimension = 12 * side**3
    matrix = np.empty((dimension, dimension))
    local_matrix = np.empty_like(matrix)
    area_matrix = np.empty_like(matrix)
    repair_matrix = np.empty_like(matrix)
    parallel_matrix = np.empty_like(matrix)
    onsite_area_quartic_matrix = np.empty_like(matrix)
    onsite_area_sextic_matrix = np.empty_like(matrix)
    curvature_matrix = np.empty_like(matrix)
    plaquette_matrix = np.empty_like(matrix)
    spectral_matrix = np.empty_like(matrix)
    basis = np.zeros(dimension)
    for column in range(dimension):
        basis[column] = 1
        dpsi, dchi = unpack(basis, side)
        local, area = tangent_force_parts(
            psi, chi, dpsi, dchi, bond_quartic, bond_sextic,
            local_cross_scale, local_self_scale)
        repair = repair_tangent_force(psi, chi, dpsi, dchi)
        parallel = parallel_tangent_force(psi, chi, dpsi, dchi)
        onsite4 = onsite_area_polynomial_tangent_force(
            psi, chi, dpsi, dchi, 2)
        onsite6 = onsite_area_polynomial_tangent_force(
            psi, chi, dpsi, dchi, 3)
        curvature = c19_curvature_tangent_force(psi, chi, dpsi, dchi)
        plaquette = plaquette_curvature_tangent_force(
            psi, chi, dpsi, dchi)
        spectral = spectral_geometry_tangent_force(psi, chi, dpsi, dchi)
        ap, ac = local[0] + area[0], local[1] + area[1]
        matrix[:, column] = pack(ap, ac)
        local_matrix[:, column] = pack(*local)
        area_matrix[:, column] = pack(*area)
        repair_matrix[:, column] = pack(*repair)
        parallel_matrix[:, column] = pack(*parallel)
        onsite_area_quartic_matrix[:, column] = pack(*onsite4)
        onsite_area_sextic_matrix[:, column] = pack(*onsite6)
        curvature_matrix[:, column] = pack(*curvature)
        plaquette_matrix[:, column] = pack(*plaquette)
        spectral_matrix[:, column] = pack(*spectral)
        basis[column] = 0
    return (matrix, local_matrix, area_matrix, repair_matrix, parallel_matrix,
            onsite_area_quartic_matrix, onsite_area_sextic_matrix,
            curvature_matrix, plaquette_matrix, spectral_matrix)


def rotation_matrix(side: int, qp: complex, qc: complex, exponent: int) -> np.ndarray:
    dimension = 12 * side**3
    result = np.empty((dimension, dimension))
    basis = np.zeros(dimension)
    pf = np.asarray((qp, qp, qp / qc)) ** exponent
    cf = np.asarray((qc, qc, 1.0 + 0.0j)) ** exponent
    for column in range(dimension):
        basis[column] = 1
        dpsi, dchi = unpack(basis, side)
        result[:, column] = pack(dpsi * pf, dchi * cf)
        basis[column] = 0
    return result


def translation_vectors(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    columns = []
    for axis in AXES:
        dp = (np.roll(psi, 1, axis=axis) - np.roll(psi, -1, axis=axis)) / 2
        dc = (np.roll(chi, 1, axis=axis) - np.roll(chi, -1, axis=axis)) / 2
        column = pack(dp, dc)
        columns.append(column / np.linalg.norm(column))
    return np.column_stack(columns)


def orth(columns: list[np.ndarray] | np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    matrix = np.column_stack(columns) if isinstance(columns, list) else columns
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, s > tolerance * max(float(s[0]), 1.0)]


def internal_mode_bases(psi: np.ndarray, chi: np.ndarray):
    zero = np.zeros_like(psi)
    global_columns = [pack(psi, zero), pack(1j * psi, zero),
                      pack(zero, chi * (norm2(psi) > 0)[..., None]),
                      pack(zero, 1j * chi * (norm2(psi) > 0)[..., None])]
    corner_columns = []
    corner_radial, corner_phase = [], []
    for site in np.argwhere(norm2(psi) > 1e-20):
        site = tuple(int(x) for x in site)
        for field, phase in (("psi", 1), ("psi", 1j),
                             ("chi", 1), ("chi", 1j)):
            dp, dc = np.zeros_like(psi), np.zeros_like(chi)
            (dp if field == "psi" else dc)[site] = phase * (
                psi[site] if field == "psi" else chi[site])
            column = pack(dp, dc)
            corner_columns.append(column)
            (corner_radial if phase == 1 else corner_phase).append(column)
    return (orth(global_columns), orth(corner_columns),
            orth(corner_radial), orth(corner_phase))


def classify(side: int, dt: float, area_scale: float,
             repair_strength: float, spectral_strength: float,
             bond_quartic: float, bond_sextic: float,
             inertia: float = 1197.0,
             parallel_strength: float = 0.0,
             local_cross_scale: float = 1.0,
             local_self_scale: float = 1.0,
             onsite_area_quartic: float = 0.0,
             onsite_area_sextic: float = 0.0,
             c19_curvature_strength: float = 0.0,
             plaquette_curvature_strength: float = 0.0) -> dict:
    global B
    B = inertia
    psi, chi, qp, qc, roots = rest_state(
        side, dt, area_scale, bond_quartic, bond_sextic, local_cross_scale,
        local_self_scale, onsite_area_quartic, onsite_area_sextic,
        c19_curvature_strength, plaquette_curvature_strength)
    ap, ac = force(psi, chi, area_scale, repair_strength, bond_quartic,
                   bond_sextic, parallel_strength, local_cross_scale,
                   local_self_scale, onsite_area_quartic,
                   onsite_area_sextic, c19_curvature_strength,
                   plaquette_curvature_strength)
    previous = (psi / qp, chi.copy())
    previous[1][..., :2] /= qc
    next_psi = 2 * psi - previous[0] + dt**2 * ap
    next_chi = 2 * chi - previous[1] + dt**2 * ac
    target_chi = chi.copy()
    target_chi[..., :2] *= qc
    orbit_defect = max(np.max(abs(next_psi - qp * psi)),
                       np.max(abs(next_chi - target_chi)))

    (Kbase, Klocal, Karea, Krepair, Kparallel, Konsite4, Konsite6, Kcurvature,
     Kplaquette, Kspectral) = tangent_matrices(
        psi, chi, bond_quartic, bond_sextic, local_cross_scale,
        local_self_scale)
    K = (Klocal + area_scale * Karea + repair_strength * Krepair
         + parallel_strength * Kparallel + onsite_area_quartic * Konsite4
         + onsite_area_sextic * Konsite6
         + c19_curvature_strength * Kcurvature
         + plaquette_curvature_strength * Kplaquette
         + spectral_strength * Kspectral)
    split_defect = np.linalg.norm(
        K - Klocal - area_scale * Karea - repair_strength * Krepair
        - parallel_strength * Kparallel
        - onsite_area_quartic * Konsite4 - onsite_area_sextic * Konsite6
        - c19_curvature_strength * Kcurvature
        - plaquette_curvature_strength * Kplaquette
        - spectral_strength * Kspectral) / np.linalg.norm(K)
    reciprocity = np.linalg.norm(K - K.T) / np.linalg.norm(K)
    rm1 = rotation_matrix(side, qp, qc, -1)
    rm2 = rotation_matrix(side, qp, qc, -2)
    identity = np.eye(K.shape[0])
    zero = np.zeros_like(K)
    monodromy = np.block([
        [rm1 @ (2 * identity + dt**2 * K), -rm2],
        [identity, zero],
    ])
    values, left_vectors, vectors = eig(monodromy, left=True, right=True,
                                        check_finite=False)
    moduli = abs(values)
    order = np.argsort(moduli)[::-1]
    translations = translation_vectors(psi, chi)
    global_internal, corner_internal, corner_radial, corner_phase = internal_mode_bases(
        psi, chi)
    leading_position = vectors[:K.shape[0], order[0]]
    leading_position /= np.linalg.norm(leading_position)
    overlap = np.sum(abs(translations.conj().T @ leading_position) ** 2)
    global_internal_overlap = np.sum(abs(global_internal.conj().T @ leading_position) ** 2)
    corner_internal_overlap = np.sum(abs(corner_internal.conj().T @ leading_position) ** 2)
    corner_radial_overlap = np.sum(abs(corner_radial.conj().T @ leading_position) ** 2)
    corner_phase_overlap = np.sum(abs(corner_phase.conj().T @ leading_position) ** 2)
    leading = vectors[:, order[0]]
    left = left_vectors[:, order[0]]
    pairing = np.vdot(left, leading)
    left = left / pairing.conjugate()
    zero_block = np.zeros_like(K)
    sensitivities = {}
    for name, part in (("local", Klocal), ("area", area_scale * Karea),
                       ("positive_density_memory", repair_strength * Krepair),
                       ("local_parallel_overlap", parallel_strength * Kparallel),
                       ("onsite_area_quartic", onsite_area_quartic * Konsite4),
                       ("onsite_area_sextic", onsite_area_sextic * Konsite6),
                       ("c19_curvature", c19_curvature_strength * Kcurvature),
                       ("plaquette_curvature",
                        plaquette_curvature_strength * Kplaquette),
                       ("spectral_geometry", spectral_strength * Kspectral)):
        derivative = np.block([
            [dt**2 * rm1 @ part, zero_block],
            [zero_block, zero_block],
        ])
        dlam = np.vdot(left, derivative @ leading)
        sensitivities[name] = {
            "eigenvalue_derivative_real": float(dlam.real),
            "eigenvalue_derivative_imag": float(dlam.imag),
            "growth_rate_derivative": float((dlam / values[order[0]]).real / dt),
        }
    shape = (side, side, side, 4, 3)
    mode = leading_position.reshape(shape)
    psi_weight = np.sum(abs(mode[..., :2, :])**2, axis=(-2, -1))
    chi_weight = np.sum(abs(mode[..., 2:, :])**2, axis=(-2, -1))
    psi_components = np.sum(abs(mode[..., :2, :])**2,
                            axis=(0, 1, 2, 3))
    chi_components = np.sum(abs(mode[..., 2:, :])**2,
                            axis=(0, 1, 2, 3))
    core = norm2(psi) > 1e-20
    total_weight = float(np.sum(psi_weight + chi_weight))
    return {
        "side": side,
        "dt": dt,
        "chi_kinetic_inertia_B": B,
        "area_scale": area_scale,
        "repair_strength": repair_strength,
        "parallel_overlap_strength": parallel_strength,
        "local_cross_density_scale": local_cross_scale,
        "local_self_potential_scale": local_self_scale,
        "onsite_area_quartic_strength": onsite_area_quartic,
        "onsite_area_sextic_strength": onsite_area_sextic,
        "c19_curvature_strength": c19_curvature_strength,
        "plaquette_curvature_strength": plaquette_curvature_strength,
        "spectral_geometry_strength": spectral_strength,
        "bond_quartic_coefficient": bond_quartic,
        "bond_sextic_coefficient": bond_sextic,
        "exact_rest_amplitude_sq": float(norm2(psi).max()),
        "positive_rest_amplitude_roots": roots,
        "configuration_real_dimension": int(K.shape[0]),
        "two_layer_real_dimension": int(monodromy.shape[0]),
        "exact_orbit_one_step_defect": float(orbit_defect),
        "mass_normalized_reciprocity_defect": float(reciprocity),
        "local_area_split_defect": float(split_defect),
        "spectral_radius": float(moduli[order[0]]),
        "leading_growth_rate_per_unit_time": float(np.log(moduli[order[0]]) / dt),
        "minimum_multiplier_modulus": float(np.min(moduli)),
        "multipliers_outside_unit_circle_1e_8": int(np.sum(moduli > 1 + 1e-8)),
        "multipliers_inside_unit_circle_1e_8": int(np.sum(moduli < 1 - 1e-8)),
        "largest_multipliers": [
            {"real": float(values[i].real), "imag": float(values[i].imag),
             "modulus": float(moduli[i])}
            for i in order[:12]
        ],
        "leading_mode_translation_overlap_sq": float(overlap.real),
        "leading_mode_global_radial_phase_overlap_sq": float(global_internal_overlap.real),
        "leading_mode_independent_corner_radial_phase_overlap_sq": float(
            corner_internal_overlap.real),
        "leading_mode_independent_corner_radial_overlap_sq": float(
            corner_radial_overlap.real),
        "leading_mode_independent_corner_phase_overlap_sq": float(
            corner_phase_overlap.real),
        "leading_mode_mass_weighted_psi_fraction": float(np.sum(psi_weight) / total_weight),
        "leading_mode_mass_weighted_psi_component_fractions": [
            float(value / total_weight) for value in psi_components],
        "leading_mode_mass_weighted_chi_component_fractions": [
            float(value / total_weight) for value in chi_components],
        "leading_mode_mass_weighted_core_fraction": float(
            np.sum((psi_weight + chi_weight)[core]) / total_weight),
        "leading_mode_action_sensitivities": sensitivities,
        "scope": "Complete two-layer tangent on the stated finite periodic box in the exact internal co-rotating frame; numerical eigenspectrum, not a Lean theorem.",
    }


def safe_classify(side: int, dt: float, area_scale: float,
                  repair_strength: float, spectral_strength: float,
                  bond_quartic: float, bond_sextic: float,
                  inertia: float, parallel_strength: float,
                  local_cross_scale: float, local_self_scale: float,
                  onsite_area_quartic: float, onsite_area_sextic: float,
                  c19_curvature_strength: float,
                  plaquette_curvature_strength: float) -> dict:
    try:
        return classify(side, dt, area_scale, repair_strength,
                        spectral_strength, bond_quartic, bond_sextic, inertia,
                        parallel_strength, local_cross_scale,
                        local_self_scale, onsite_area_quartic,
                        onsite_area_sextic, c19_curvature_strength,
                        plaquette_curvature_strength)
    except ValueError as error:
        return {
            "side": side,
            "dt": dt,
            "chi_kinetic_inertia_B": inertia,
            "area_scale": area_scale,
            "repair_strength": repair_strength,
            "parallel_overlap_strength": parallel_strength,
            "local_cross_density_scale": local_cross_scale,
            "local_self_potential_scale": local_self_scale,
            "onsite_area_quartic_strength": onsite_area_quartic,
            "onsite_area_sextic_strength": onsite_area_sextic,
            "c19_curvature_strength": c19_curvature_strength,
            "plaquette_curvature_strength": plaquette_curvature_strength,
            "spectral_geometry_strength": spectral_strength,
            "bond_quartic_coefficient": bond_quartic,
            "bond_sextic_coefficient": bond_sextic,
            "classification_error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", nargs="+", type=int, default=(3, 4))
    parser.add_argument("--rates", nargs="+", type=int, default=(128, 256, 512))
    parser.add_argument("--strengths", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--area-scales", nargs="+", type=float, default=(1.0,))
    parser.add_argument("--spectral-strengths", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--bond-quartics", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--bond-sextics", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--inertias", nargs="+", type=float, default=(1197.0,))
    parser.add_argument("--parallel-strengths", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--local-cross-scales", nargs="+", type=float,
                        default=(1.0,))
    parser.add_argument("--local-self-scales", nargs="+", type=float,
                        default=(1.0,))
    parser.add_argument("--onsite-area-quartics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--onsite-area-sextics", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--c19-curvature-strengths", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--plaquette-curvature-strengths", nargs="+", type=float,
                        default=(0.0,))
    parser.add_argument("--output", default="localized_stability.json")
    args = parser.parse_args()
    rows = [safe_classify(side, 1 / rate, area_scale, strength, spectral_strength,
                          bond_quartic, bond_sextic, inertia,
                          parallel_strength, local_cross_scale,
                          local_self_scale, onsite_area_quartic,
                          onsite_area_sextic, c19_curvature_strength,
                          plaquette_curvature_strength)
            for side in args.sides for rate in args.rates
            for area_scale in args.area_scales for strength in args.strengths
            for spectral_strength in args.spectral_strengths
            for bond_quartic in args.bond_quartics
            for bond_sextic in args.bond_sextics
            for inertia in args.inertias
            for parallel_strength in args.parallel_strengths
            for local_cross_scale in args.local_cross_scales
            for local_self_scale in args.local_self_scales
            for onsite_area_quartic in args.onsite_area_quartics
            for onsite_area_sextic in args.onsite_area_sextics
            for c19_curvature_strength in args.c19_curvature_strengths
            for plaquette_curvature_strength in args.plaquette_curvature_strengths]
    result = {
        "status": "NUMERICAL_FULL_TANGENT_CLASSIFICATION",
        "action_changed": any(
            value != baseline
            for values, baseline in (
                (args.strengths, 0.0), (args.area_scales, 1.0),
                (args.spectral_strengths, 0.0), (args.bond_quartics, 0.0),
                (args.bond_sextics, 0.0), (args.inertias, 1197.0))
            for value in values) or any(
                value != 0.0 for value in args.parallel_strengths) or any(
                value != 1.0 for value in args.local_cross_scales) or any(
                value != 1.0 for value in args.local_self_scales) or any(
                value != 0.0 for value in args.onsite_area_quartics) or any(
                value != 0.0 for value in args.onsite_area_sextics) or any(
                value != 0.0 for value in args.c19_curvature_strengths) or any(
                value != 0.0 for value in args.plaquette_curvature_strengths),
        "rows": rows,
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
