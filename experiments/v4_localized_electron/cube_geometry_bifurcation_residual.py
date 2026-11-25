"""Classify the mode left by the cube-geometric bifurcation solve.

This diagnostic reconstructs the complete native two-layer spectrum at the
continuous inverse solution.  It does not admit those fitted coefficients.
It resolves the fastest surviving mode by field, region, C3 component and
the exact eight-corner Walsh characters of the carrier cube.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import current_action_localized_envelope_closure as envelope
import current_action_two_layer_spectrum as current
import cube_geometry_bifurcation_solve as solve
import density_connection_residual_mode as classify
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5


def assemble(coefficients: np.ndarray):
    bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    blocks = [solve.load_blocks(feature) for feature in solve.FEATURES]
    natural = native_time.B*native_time.CHI0
    matrices = {
        "mass": mass0.copy(),
        "gyro": gyro0.copy(),
        "stiffness": stiffness0.copy(),
    }
    for coefficient, delta in zip(coefficients, blocks, strict=True):
        for kind in matrices:
            matrices[kind] += natural*coefficient*delta[kind]
    return bare, blocks, matrices


def fastest_mode(bare, matrices):
    mass = matrices["mass"]
    gyro = matrices["gyro"]
    stiffness = matrices["stiffness"]
    dressed = current.dressed_configuration(stiffness, mass, bare["rest"])
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(factor, (dt*stiffness-2*mass/dt)@dressed)
    backward = -lu_solve(factor, (mass/dt-gyro/2)@dressed)
    residual = np.column_stack((
        forward-dressed@(dressed.T@forward),
        backward-dressed@(dressed.T@backward)))
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    extra = u[:, singular > 1e-10*max(float(singular[0]), 1.0)]
    hull = envelope.orth(np.column_stack((dressed, extra)))
    fh = -lu_solve(factor, (dt*stiffness-2*mass/dt)@hull)
    bh = -lu_solve(factor, (mass/dt-gyro/2)@hull)
    identity = np.eye(hull.shape[1])
    monodromy = np.block([
        [hull.T@fh, hull.T@bh],
        [identity, np.zeros_like(identity)],
    ])
    values, vectors = eig(monodromy, check_finite=False)
    index = int(np.argmax(abs(values)))
    present = hull@vectors[:hull.shape[1], index]
    return values[index], present


def corner_characters(present: np.ndarray, rest: np.ndarray):
    psi, chi = native.unpack(rest, SIDE)
    dpsi, dchi = native.unpack(present, SIDE)
    _pf, _cf, sites = envelope.carrier_frames(rest)
    unique = [sorted({site[axis] for site in sites}) for axis in range(3)]
    bits = [tuple(values.index(site[axis])
                  for axis, values in enumerate(unique)) for site in sites]
    rows = {}
    for field_name, field, delta in (
            ("Psi", psi, dpsi), ("chi", chi, dchi)):
        amplitudes = []
        for site in sites:
            norm = max(float(np.vdot(field[site], field[site]).real), 1e-30)
            amplitudes.append(np.vdot(field[site], delta[site])/norm)
        amplitudes = np.asarray(amplitudes)
        total = max(float(np.vdot(amplitudes, amplitudes).real), 1e-30)
        characters = {}
        for parity in itertools.product((0, 1), repeat=3):
            vector = np.asarray([
                (-1.0)**sum(parity[axis]*bit[axis] for axis in range(3))
                for bit in bits])/np.sqrt(8)
            coefficient = np.vdot(vector, amplitudes)
            label = "".join(str(value) for value in parity)
            characters[label] = {
                "complex_coefficient": [float(coefficient.real),
                                          float(coefficient.imag)],
                "power_fraction": float(abs(coefficient)**2/total),
            }
        rows[field_name] = characters

    adjacency = np.zeros((8, 8))
    for left, site_i in enumerate(sites):
        for right, site_j in enumerate(sites):
            delta = tuple(min((site_j[a]-site_i[a]) % SIDE,
                              (site_i[a]-site_j[a]) % SIDE)
                          for a in range(3))
            if sum(delta) == 1:
                adjacency[left, right] = 2
            elif sum(delta) == 2 and max(delta) <= 1:
                adjacency[left, right] = 1
    laplacian = np.diag(adjacency.sum(axis=1))-adjacency
    for parity in itertools.product((0, 1), repeat=3):
        vector = np.asarray([
            (-1.0)**sum(parity[axis]*bit[axis] for axis in range(3))
            for bit in bits])/np.sqrt(8)
        label = "".join(str(value) for value in parity)
        value = float(vector@laplacian@vector)
        for field_name in rows:
            rows[field_name][label]["restricted_C19_eigenvalue"] = value
    return rows


def main() -> None:
    fit = json.loads((HERE/"cube_geometry_bifurcation_solve.json").read_text())
    coefficients = np.asarray(
        fit["continuous_solution"]["coefficients_in_B_chi0_units"])
    bare, blocks, matrices = assemble(coefficients)
    multiplier, present = fastest_mode(bare, matrices)
    shaped = present.reshape(SIDE, SIDE, SIDE, 4, 3)
    total = float(np.sum(abs(shaped)**2))
    regions = classify.spatial_regions(bare["rest"])
    field_power = np.sum(abs(shaped)**2, axis=(0, 1, 2, 4))/total
    component_power = np.sum(abs(shaped)**2, axis=(0, 1, 2, 3))/total
    mass_norm = max(float(abs(np.vdot(present, matrices["mass"]@present))),
                    1e-30)
    sensitivities = {}
    for feature, delta in zip(solve.FEATURES, blocks, strict=True):
        sensitivities[feature] = {
            kind: float(np.vdot(present, delta[kind]@present).real/mass_norm)
            for kind in ("mass", "gyro", "stiffness")
        }
    characters = corner_characters(present, bare["rest"])
    dominant = {
        field: max(rows, key=lambda name: rows[name]["power_fraction"])
        for field, rows in characters.items()
    }
    result = {
        "status": "V4_CUBE_GEOMETRY_BIFURCATION_RESIDUAL",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficients_in_B_chi0_units": coefficients.tolist(),
        "multiplier": {
            "real": float(multiplier.real),
            "imag": float(multiplier.imag),
            "modulus": float(abs(multiplier)),
            "growth_rate": float(np.log(abs(multiplier))/native_time.DT),
        },
        "field_power_fractions": {
            "Psi_real": float(field_power[0]),
            "Psi_imag": float(field_power[1]),
            "chi_real": float(field_power[2]),
            "chi_imag": float(field_power[3]),
        },
        "C3_component_power_fractions": component_power.tolist(),
        "spatial_region_power_fractions": {
            name: classify.region_power(shaped, mask)/total
            for name, mask in regions.items()
        },
        "dominant_carrier_cube_characters": dominant,
        "carrier_cube_characters": characters,
        "unit_feature_quadratics": sensitivities,
        "verdict": "TESTED_SCALAR_CONNECTION_SCREEN_RETAINS_GROWTH",
        "claim_boundary": (
            "This is a mode classification at an inverse fitted point. It "
            "does not admit the coefficients or prove a no-go outside the "
            "six declared reciprocal scalar-connection channels."),
    }
    (HERE/"cube_geometry_bifurcation_residual.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
