"""Resolve the natural-connection residual into eight-corner characters."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import current_action_localized_envelope_closure as envelope
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def physical_projection(vector: np.ndarray, phase: float,
                        psi: np.ndarray, sites: list[tuple[int, ...]]):
    packed = np.real(np.exp(1j*phase)*vector)
    dpsi, _dchi = native.unpack(packed, SIDE)
    radial, angular = [], []
    reconstructed = np.zeros_like(dpsi)
    for site in sites:
        rho = float(native.norm2(psi)[site])
        overlap = np.vdot(psi[site], dpsi[site])/rho
        radial.append(float(overlap.real))
        angular.append(float(overlap.imag))
        reconstructed[site] = overlap*psi[site]
    fraction = float(
        np.sum(abs(reconstructed)**2)/max(np.sum(abs(dpsi)**2), 1e-30))
    phase_fraction = float(
        np.sum(np.asarray(angular)**2)/max(
            np.sum(np.asarray(radial)**2)+np.sum(np.asarray(angular)**2),
            1e-30))
    return np.asarray(radial), np.asarray(angular), fraction, phase_fraction


def main() -> None:
    bare = np.load(canonical.CACHE)
    residual = np.load(HERE/"density_connection_fastest_residual_mode.npz")
    vector = residual["current"]
    psi, _chi = native.unpack(bare["rest"], SIDE)
    _psi_frame, _chi_frame, sites = envelope.carrier_frames(bare["rest"])

    candidates = []
    for phase in np.linspace(0, 2*np.pi, 1441, endpoint=False):
        radial, angular, fraction, phase_fraction = physical_projection(
            vector, float(phase), psi, sites)
        candidates.append((phase_fraction, fraction, phase, radial, angular))
    phase_fraction, fraction, selected_phase, radial, angular = max(
        candidates, key=lambda row: (row[0], row[1]))

    unique = [sorted({site[axis] for site in sites}) for axis in range(3)]
    bits = [tuple(values.index(site[axis]) for axis, values in enumerate(unique))
            for site in sites]
    characters = {}
    matrix_columns = []
    labels = []
    for parity in itertools.product((0, 1), repeat=3):
        values = np.asarray([
            (-1.0)**sum(parity[axis]*bit[axis] for axis in range(3))
            for bit in bits]) / np.sqrt(8)
        label = "".join(str(value) for value in parity)
        labels.append(label)
        matrix_columns.append(values)
        characters[label] = {
            "coefficient": float(np.dot(values, angular)),
            "power_fraction": float(
                np.dot(values, angular)**2/max(np.dot(angular, angular), 1e-30)),
        }
    character_matrix = np.column_stack(matrix_columns)

    # C19 adjacency restricted to the eight carrier sites.
    adjacency = np.zeros((8, 8))
    for left, site_i in enumerate(sites):
        for right, site_j in enumerate(sites):
            delta = tuple(min((site_j[a]-site_i[a]) % SIDE,
                              (site_i[a]-site_j[a]) % SIDE)
                          for a in range(3))
            manhattan = sum(delta)
            if manhattan == 1:
                adjacency[left, right] = 2
            elif manhattan == 2 and max(delta) <= 1:
                adjacency[left, right] = 1
    laplacian = np.diag(np.sum(adjacency, axis=1))-adjacency
    graph_eigenvalues = np.diag(character_matrix.T@laplacian@character_matrix)

    order = sorted(labels, key=lambda name: characters[name]["power_fraction"],
                   reverse=True)
    result = {
        "status": "V4_DENSITY_CONNECTION_RESIDUAL_CORNER_ALGEBRA",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "selected_real_section_phase": float(selected_phase),
        "ray_projection_fraction": fraction,
        "phase_fraction_within_ray_projection": phase_fraction,
        "sites": [list(site) for site in sites],
        "cube_bits": [list(bit) for bit in bits],
        "normalized_radial_coefficients": (
            radial/max(np.linalg.norm(radial), 1e-30)).tolist(),
        "normalized_phase_coefficients": (
            angular/max(np.linalg.norm(angular), 1e-30)).tolist(),
        "corner_character_decomposition": {
            name: {
                **characters[name],
                "restricted_C19_graph_laplacian_eigenvalue": float(
                    graph_eigenvalues[labels.index(name)]),
            } for name in order
        },
        "dominant_character": order[0],
        "dominant_character_power_fraction": characters[order[0]][
            "power_fraction"],
        "verdict": "RESIDUAL_CORNER_CHARACTER_IDENTIFIED",
        "claim_boundary": (
            "This is an observational decomposition of the fastest residual, "
            "not a runtime parity selector or an admissible action term."),
    }
    (HERE/"density_connection_residual_corner_algebra.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
