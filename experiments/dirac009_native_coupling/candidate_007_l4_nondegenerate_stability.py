"""Nondegenerate L=4 validation of the DIRAC-009 incidence action.

The old eight-site test identifies +axis and -axis neighbours and therefore
makes every centered incidence residual zero.  This test repeats the prepared
eight-ray background on a genuine 4x4x4 periodic lattice, where those
neighbours are distinct.  Both complex C3 fields evolve freely under the
complete declared positive action.  No projection or held background occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "candidate_007_l4_nondegenerate_stability.json"
L = 4
N = L**3
PERTURBATION = 1.0e-4
CYCLES = 100
CHI0 = 19.0
B = 1197.0
AMPLITUDE = 8.111071056538126
CHI_RADIUS = 17.18169160227797
OMEGA = 19.0


def tetra_lines() -> np.ndarray:
    a = 1.0 / np.sqrt(3.0)
    b = np.sqrt(2.0 / 3.0)
    omega = -0.5 + 0.5j * np.sqrt(3.0)
    rays = np.asarray([
        [1.0, 0.0, 0.0],
        [a, b, 0.0],
        [a, b * omega, 0.0],
        [a, b * np.conj(omega), 0.0],
    ], dtype=np.complex128)
    return np.asarray([
        rays[0], rays[1], rays[2], rays[3],
        -rays[3], -rays[2], -rays[1], -rays[0],
    ])


def companion(lines: np.ndarray) -> np.ndarray:
    result = np.stack((
        -np.conj(lines[:, 1]), np.conj(lines[:, 0]),
        np.zeros(8, dtype=np.complex128),
    ), axis=1)
    return result / np.linalg.norm(result, axis=1)[:, None]


def corner_index(residue: np.ndarray) -> int:
    r = np.mod(residue, 2)
    return int(4 * r[0] + 2 * r[1] + r[2])


def site_index(x: int, y: int, z: int) -> int:
    return ((x % L) * L + (y % L)) * L + (z % L)


def geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets: list[tuple[int, int, int]] = []
    weights: list[float] = []
    face_signs: list[float] = []
    for axis in range(3):
        for sign in (1, -1):
            offset = [0, 0, 0]
            offset[axis] = sign
            offsets.append(tuple(offset))
            weights.append(2.0)
            face_signs.append(float(sign))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for sign_first in (-1, 1):
            for sign_second in (-1, 1):
                offset = [0, 0, 0]
                offset[first] = sign_first
                offset[second] = sign_second
                offsets.append(tuple(offset))
                weights.append(1.0)
                face_signs.append(0.0)
    neighbors = np.zeros((N, 18), dtype=np.int64)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                source = site_index(x, y, z)
                for direction, (dx, dy, dz) in enumerate(offsets):
                    neighbors[source, direction] = site_index(x + dx, y + dy, z + dz)
    return (
        torch.tensor(neighbors, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(face_signs, dtype=torch.float64),
        torch.tensor(offsets, dtype=torch.int64),
    )


NEIGHBORS, WEIGHTS, FACE_SIGNS, OFFSETS = geometry()


def wedge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
        left[..., 0] * right[..., 2] - left[..., 2] * right[..., 0],
        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
    ), dim=-1)


def complex_fields(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sites = q.reshape(N, 12)
    psi_parts = sites[:, :6].reshape(N, 3, 2)
    chi_parts = sites[:, 6:].reshape(N, 3, 2)
    return (
        torch.complex(psi_parts[..., 0], psi_parts[..., 1]),
        torch.complex(chi_parts[..., 0], chi_parts[..., 1]),
    )


def real_norm_squared(value: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.sum(torch.real(torch.conj(value) * value), dim=dim)


def action_terms(q: torch.Tensor) -> dict[str, torch.Tensor]:
    psi, chi = complex_fields(q)
    rho_psi = real_norm_squared(psi, dim=1)
    rho_chi = real_norm_squared(chi, dim=1)
    local = torch.sum(
        0.25 * (rho_chi - CHI0**2) ** 2
        + 0.5 * rho_chi * rho_psi
        + 0.25 * rho_psi**2
    )

    track = wedge(chi, psi)
    track_neighbors = track[NEIGHBORS]
    bond_residual = track[:, None, :] - track_neighbors
    bond = torch.sum(
        WEIGHTS[None, :, None]
        * torch.real(torch.conj(bond_residual) * bond_residual)
    ) / (2.0 * CHI0**2)

    chi_neighbors = chi[NEIGHBORS]
    outer = chi_neighbors[..., :, None] * torch.conj(chi_neighbors[..., None, :])
    covariance = torch.sum(
        WEIGHTS[None, :, None, None] * outer, dim=1
    ) / (8.0 * CHI0**2)
    covariance_chi = torch.einsum("nab,nb->na", covariance, chi)
    shape_residual = wedge(chi, covariance_chi)
    shape = 0.5 * real_norm_squared(shape_residual)

    overlaps = torch.sum(torch.conj(chi[:, None, :]) * chi_neighbors, dim=2)
    incidence_residual = torch.sum(
        (3.0 * FACE_SIGNS[None, :, None] / (2.0 * CHI0**2))
        * overlaps[..., None] * track_neighbors,
        dim=1,
    )
    incidence = 0.5 * real_norm_squared(incidence_residual)
    return {
        "local": local,
        "bond": bond,
        "shape": shape,
        "incidence": incidence,
        "total": local + bond + shape + incidence,
        "incidence_residual_norm": torch.linalg.vector_norm(incidence_residual),
    }


def potential(q: torch.Tensor) -> torch.Tensor:
    return action_terms(q)["total"]


def acceleration(q: torch.Tensor, inverse_mass: torch.Tensor) -> torch.Tensor:
    point = q.detach().requires_grad_(True)
    gradient = torch.autograd.grad(potential(point), point)[0]
    return (-inverse_mass * gradient).detach()


def pack_site(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    row = np.zeros(12, dtype=np.float64)
    for component in range(3):
        row[2 * component] = psi[component].real
        row[2 * component + 1] = psi[component].imag
        row[6 + 2 * component] = chi[component].real
        row[7 + 2 * component] = chi[component].imag
    return row


def initial_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rays = tetra_lines()
    companions = companion(rays)
    psi_reference = np.zeros((N, 3), dtype=np.complex128)
    chi_reference = np.zeros((N, 3), dtype=np.complex128)
    rows = np.zeros((N, 12), dtype=np.float64)
    velocity = np.zeros((N, 12), dtype=np.float64)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                site = site_index(x, y, z)
                corner = corner_index(np.asarray([x % 2, y % 2, z % 2]))
                psi = AMPLITUDE * rays[corner]
                chi = CHI_RADIUS * companions[corner]
                psi_reference[site] = psi
                chi_reference[site] = chi
                rows[site] = pack_site(psi, chi)
                for component in range(3):
                    velocity[site, 2 * component] = -OMEGA * psi[component].imag
                    velocity[site, 2 * component + 1] = OMEGA * psi[component].real
    return rows.reshape(-1), velocity.reshape(-1), psi_reference, chi_reference


def unpack(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = q.reshape(N, 12)
    psi = rows[:, :6].reshape(N, 3, 2)
    chi = rows[:, 6:].reshape(N, 3, 2)
    return psi[..., 0] + 1j * psi[..., 1], chi[..., 0] + 1j * chi[..., 1]


def diagnostics(q: np.ndarray, psi_ref: np.ndarray, chi_ref: np.ndarray) -> dict[str, float]:
    psi, chi = unpack(q)
    overlap = np.vdot(psi_ref.reshape(-1), psi.reshape(-1))
    phase = np.exp(-1j * np.angle(overlap)) if abs(overlap) else 1.0
    psi_aligned = phase * psi
    psi_defect = np.linalg.norm(psi_aligned - psi_ref) / np.linalg.norm(psi_ref)
    chi_defect = np.linalg.norm(chi - chi_ref) / np.linalg.norm(chi_ref)
    track = np.cross(chi, psi)
    track_ref = np.cross(chi_ref, psi_ref)
    track_overlap = np.vdot(track_ref.reshape(-1), track.reshape(-1))
    track_phase = np.exp(-1j * np.angle(track_overlap)) if abs(track_overlap) else 1.0
    track_defect = np.linalg.norm(track_phase * track - track_ref) / np.linalg.norm(track_ref)
    return {
        "psi_pattern_defect": float(psi_defect),
        "chi_pattern_defect": float(chi_defect),
        "track_pattern_defect": float(track_defect),
    }


def run(updates_per_cycle: int) -> dict[str, object]:
    q0, v0, psi_ref, chi_ref = initial_state()
    mass_np = np.ones_like(q0)
    for site in range(N):
        mass_np[12 * site + 6:12 * (site + 1)] = B
    rng = np.random.default_rng(120009)
    perturb = rng.normal(size=q0.size)
    perturb /= np.linalg.norm(perturb)
    q = torch.tensor(
        q0 + PERTURBATION * np.linalg.norm(q0) * perturb,
        dtype=torch.float64,
    )
    velocity_perturb = rng.normal(size=q0.size) / np.sqrt(mass_np)
    velocity_perturb /= np.sqrt(np.sum(mass_np * velocity_perturb**2))
    reference_speed = np.sqrt(np.sum(mass_np * v0**2))
    velocity = torch.tensor(
        v0 + PERTURBATION * reference_speed * velocity_perturb,
        dtype=torch.float64,
    )
    mass = torch.tensor(mass_np, dtype=torch.float64)
    inverse_mass = 1.0 / mass
    dt = (2.0 * np.pi / OMEGA) / updates_per_cycle

    def energy() -> float:
        return float((0.5 * torch.sum(mass * velocity**2) + potential(q)).detach())

    initial_energy = energy()
    initial_terms = action_terms(q)
    initial_incidence = float(initial_terms["incidence"].detach())
    initial_residual = float(initial_terms["incidence_residual_norm"].detach())
    maxima = diagnostics(q.numpy(), psi_ref, chi_ref)
    maximum_relative_energy_deviation = 0.0
    accel = acceleration(q, inverse_mass)
    for _step in range(1, CYCLES * updates_per_cycle + 1):
        velocity_half = velocity + 0.5 * dt * accel
        q = q + dt * velocity_half
        next_accel = acceleration(q, inverse_mass)
        velocity = velocity_half + 0.5 * dt * next_accel
        accel = next_accel
        row = diagnostics(q.numpy(), psi_ref, chi_ref)
        maxima = {key: max(maxima[key], row[key]) for key in maxima}
        relative_energy_deviation = (
            abs(energy() - initial_energy) / max(1.0, abs(initial_energy))
        )
        maximum_relative_energy_deviation = max(
            maximum_relative_energy_deviation, relative_energy_deviation
        )
    passed = bool(
        initial_incidence > 1.0e-14
        and initial_residual > 1.0e-10
        and maxima["psi_pattern_defect"] < 0.01
        and maxima["chi_pattern_defect"] < 0.01
        and maxima["track_pattern_defect"] < 0.01
        and maximum_relative_energy_deviation < 1.0e-4
    )
    return {
        "updates_per_cycle": updates_per_cycle,
        "dt": dt,
        "initial_incidence_energy": initial_incidence,
        "initial_incidence_residual_norm": initial_residual,
        "maximums": maxima,
        "maximum_relative_energy_deviation": maximum_relative_energy_deviation,
        "passes": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", default="64,128,256")
    args = parser.parse_args()
    updates = [int(item) for item in args.updates.split(",")]
    torch.set_default_dtype(torch.float64)
    background, _, _, _ = initial_state()
    background_terms = action_terms(torch.tensor(background, dtype=torch.float64))
    rows = [run(value) for value in updates]
    passed = all(row["passes"] for row in rows)
    payload = {
        "candidate": "DIRAC009_L4_NONDEGENERATE_INCIDENCE_ACTION",
        "lattice": "4x4x4 periodic C19",
        "cycles": CYCLES,
        "relative_perturbation": PERTURBATION,
        "background_incidence_residual_norm": float(
            background_terms["incidence_residual_norm"].detach()
        ),
        "runs": rows,
        "verdict": "PASS" if passed else "FAIL",
        "claim_boundary": (
            "Tests the full declared action on a nondegenerate repeated lattice. "
            "It establishes numerical stability of the prepared background, not "
            "a localized electron, physical mass, spin or the continuum Dirac equation."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
