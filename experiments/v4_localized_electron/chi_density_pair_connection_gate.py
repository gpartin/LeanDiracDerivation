"""Full spectrum of Psi- and chi-density-oriented pair connections.

The first connection used the Psi density contrast.  The residual inverse
screen identifies chi density as an independent rest-preserving response.
This file derives its full discrete Euler--Lagrange blocks and tests only the
native sum/difference combinations, with and without the existing chi inertia
normalization.  No coefficient is fitted to a target spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import complete_pair_wave_action_gate as spectrum
import current_action_two_layer_spectrum as current
import density_oriented_pair_connection as psi_connection
import density_pair_connection_basis_screen as basis
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def derive_chi_blocks(rest: np.ndarray, dt: float):
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]

    def residual(left, middle, right):
        action = (basis.one_form(left, middle, "rho_chi")
                  + basis.one_form(middle, right, "rho_chi"))
        return torch.autograd.grad(action, middle, create_graph=True)[0]

    matrices = []
    for argument in range(3):
        def function(value, selected=argument):
            trial = layers.copy()
            trial[selected] = value
            return residual(*trial)
        matrices.append(torch.autograd.functional.jacobian(
            function, layers[argument], vectorize=True).detach().numpy())
    minus = matrices[0]@rotation.T
    zero = matrices[1]
    plus = matrices[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    psi_cache = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    psi = {kind: psi_cache[f"{kind}_common_pair_connection"]
           for kind in ("mass", "gyro", "stiffness")}
    chi_path = HERE/"chi_density_pair_connection_matrices.npz"
    if chi_path.exists():
        cached = np.load(chi_path)
        chi = {kind: cached[kind] for kind in ("mass", "gyro", "stiffness")}
    else:
        chi = derive_chi_blocks(bare["rest"], native_time.DT)
        np.savez_compressed(chi_path, **chi)

    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    base_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    natural = native_time.B*native_time.CHI0
    combinations = (
        ("Psi_density", natural, 0.0),
        ("chi_density", 0.0, natural),
        ("total_density", natural, natural),
        ("density_difference", natural, -natural),
        ("inertia_normalized_total", natural, natural/native_time.B),
        ("inertia_normalized_difference", natural, -natural/native_time.B),
    )
    rows = []
    for label, psi_coefficient, chi_coefficient in combinations:
        mass = (bare["mass"]+psi_coefficient*psi["mass"]
                + chi_coefficient*chi["mass"])
        gyro = (bare["gyro"]+psi_coefficient*psi["gyro"]
                + chi_coefficient*chi["gyro"])
        stiffness = (base_stiffness+psi_coefficient*psi["stiffness"]
                     + chi_coefficient*chi["stiffness"])
        row = {
            "label": label,
            "Psi_density_coefficient": psi_coefficient,
            "chi_density_coefficient": chi_coefficient,
            "minimum_mass_eigenvalue": float(eigvalsh(
                mass, subset_by_index=[0, 0], driver="evr")[0]),
        }
        if row["minimum_mass_eigenvalue"] > 1e-9:
            dressed = current.dressed_configuration(
                stiffness, mass, bare["rest"])
            row.update(spectrum.spectrum_row(
                mass, gyro, stiffness, dressed, native_time.DT))
        rows.append(row)
    result = {
        "status": "V4_CHI_DENSITY_PAIR_CONNECTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "natural_scale": natural,
        "natural_scale_provenance": "B times chi0",
        "combinations": rows,
        "verdict": (
            "DENSITY_CONNECTION_COMBINATION_FULL_SPECTRUM_PASS"
            if any(row.get("growing_multiplier_count_at_1e_minus_7") == 0
                   for row in rows)
            else "DENSITY_CONNECTION_COMBINATIONS_RETAIN_GROWTH"),
        "claim_boundary": (
            "These are full finite-hull spectra of native sum/difference "
            "connections. No coefficient is fitted, no action is installed, "
            "and a pass would still require refinements and nonlinear motion."),
    }
    (HERE/"chi_density_pair_connection_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
