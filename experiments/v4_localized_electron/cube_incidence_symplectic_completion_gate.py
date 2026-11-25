"""Full-spectrum gate for the exact cube-incidence symplectic completion.

The preceding rest-force nullspace derives the integer combination

  A_cube = A_area,C19 - 4 A_tensor+,local
           + 4 A_tensor-,local - 4 A_area+,local.

No corner, direction, spinor copy or Dirac matrix occurs in these native
one-forms.  This script derives the complete DEL matrices, screens native
geometric coefficient scales in a reduced invariant space, and validates the
best candidates against the complete 2064-dimensional two-layer spectrum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eig, eigvalsh
from scipy.optimize import minimize_scalar

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import complete_pair_wave_action_gate as spectrum
import cube_geometry_bifurcation_solve as solve
import cube_incidence_symplectic_basis as basis
import current_action_two_layer_spectrum as current
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
CACHE = HERE/"cube_incidence_symplectic_completion_matrices.npz"
COEFFICIENTS = np.asarray((-4.0, 0.0, 4.0, 0.0, -4.0, 1.0, 0.0, 0.0))


def derive_blocks(rest: np.ndarray):
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]
    coefficients = torch.tensor(COEFFICIENTS, dtype=torch.float64)

    def residual(left, middle, right):
        return basis.del_residual(left, middle, right, coefficients)

    jacobians = []
    for selected in range(3):
        def function(value, argument=selected):
            trial = layers.copy()
            trial[argument] = value
            return residual(*trial)
        jacobians.append(torch.autograd.functional.jacobian(
            function, layers[selected], vectorize=True).detach().numpy())
    minus = jacobians[0]@rotation.T
    zero = jacobians[1]
    plus = jacobians[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
    }


def projected_row(scale: float, base, delta, q):
    matrices = {kind: q.T@(base[kind]+scale*delta[kind])@q
                for kind in ("mass", "gyro", "stiffness")}
    minimum_mass = float(eigvalsh(matrices["mass"])[0])
    row = {"scale": float(scale),
           "minimum_reduced_mass_eigenvalue": minimum_mass}
    if minimum_mass > 1e-9:
        multipliers = eig(solve.reduced_monodromy(**matrices),
                          right=False, check_finite=False)
        row.update({
            "maximum_multiplier_modulus": float(max(abs(multipliers))),
            "maximum_growth_rate": float(
                np.log(max(abs(multipliers)))/native_time.DT),
            "growing_multiplier_count": int(np.sum(
                abs(multipliers) > 1+1e-7)),
        })
    return row


def full_row(scale: float, bare, base, delta):
    matrices = {kind: base[kind]+scale*delta[kind]
                for kind in ("mass", "gyro", "stiffness")}
    minimum_mass = float(eigvalsh(
        matrices["mass"], subset_by_index=[0, 0], driver="evr")[0])
    row = {"scale": float(scale),
           "minimum_full_mass_eigenvalue": minimum_mass}
    if minimum_mass > 1e-9:
        try:
            dressed = current.dressed_configuration(
                matrices["stiffness"], matrices["mass"], bare["rest"])
            row.update(spectrum.spectrum_row(
                matrices["mass"], matrices["gyro"], matrices["stiffness"],
                dressed, native_time.DT))
        except (np.linalg.LinAlgError, ValueError) as error:
            row["full_validation_error"] = type(error).__name__
            try:
                multipliers = eig(solve.reduced_monodromy(**matrices),
                                  right=False, check_finite=False)
                row.update({
                    "identity_full_spectrum_fallback": True,
                    "configuration_hull_rank": int(matrices["mass"].shape[0]),
                    "phase_space_dimension": int(2*matrices["mass"].shape[0]),
                    "hull_relative_cross_block": 0.0,
                    "maximum_multiplier_modulus": float(max(abs(multipliers))),
                    "maximum_growth_rate": float(
                        np.log(max(abs(multipliers)))/native_time.DT),
                    "growing_multiplier_count_at_1e_minus_7": int(np.sum(
                        abs(multipliers) > 1+1e-7)),
                })
            except (np.linalg.LinAlgError, ValueError) as fallback_error:
                row["identity_full_spectrum_fallback_error"] = type(
                    fallback_error).__name__
    return row


def main() -> None:
    bare = np.load(canonical.CACHE)
    if CACHE.exists():
        cached = np.load(CACHE)
        delta = {kind: cached[kind]
                 for kind in ("mass", "gyro", "stiffness")}
    else:
        delta = derive_blocks(bare["rest"])
        np.savez_compressed(CACHE, **delta)

    _bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    connection = solve.load_blocks("rho_Psi")
    natural = native_time.B*native_time.CHI0
    base = {
        "mass": mass0+natural*connection["mass"],
        "gyro": gyro0+natural*connection["gyro"],
        "stiffness": stiffness0+natural*connection["stiffness"],
    }
    hull = solve.invariant_hull(**base, rest=bare["rest"])
    values, growing, residual = solve.real_growing_configuration(
        hull, solve.monodromy(**base, q=hull))
    response_space = solve.envelope.orth(np.column_stack((
        residual,
        delta["mass"]@residual,
        delta["gyro"]@residual,
        delta["stiffness"]@residual,
    )))

    candidates = (0.0, 1/native_time.CHI0**2,
                  -1/native_time.CHI0**2,
                  1/native_time.CHI0, -1/native_time.CHI0,
                  0.25, -0.25, 1.0, -1.0, 4.0, -4.0,
                  6.0, -6.0, 8.0, 12.0,
                  native_time.CHI0, -native_time.CHI0)
    projected = [projected_row(value, base, delta, response_space)
                 for value in candidates]

    def objective(value: float):
        row = projected_row(value, base, delta, response_space)
        return row.get("maximum_growth_rate", 1000.0)

    diagnostic = minimize_scalar(
        objective, bounds=(-native_time.CHI0, native_time.CHI0),
        method="bounded", options={"xatol": 1e-10, "maxiter": 300})
    projected.append({
        **projected_row(float(diagnostic.x), base, delta, response_space),
        "diagnostic_continuous_inverse": True,
        "optimizer_success": bool(diagnostic.success),
    })
    projected.sort(key=lambda row: row.get("maximum_growth_rate", np.inf))
    full_scales = []
    for row in projected[:9]:
        if row["scale"] not in full_scales:
            full_scales.append(row["scale"])
    full = [full_row(value, bare, base, delta) for value in full_scales]
    finite_full = [row for row in full if "maximum_growth_rate" in row]
    best = min(finite_full,
               key=lambda row: row.get("maximum_growth_rate", np.inf))

    coefficients = torch.tensor(COEFFICIENTS, dtype=torch.float64)
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True)
                   for value in (rotation.T@bare["rest"], bare["rest"],
                                 rotation@bare["rest"]))
    rest_del = basis.del_residual(*layers, coefficients).detach().numpy()
    result = {
        "status": "V4_CUBE_INCIDENCE_SYMPLECTIC_COMPLETION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis_coefficients": {
            name: float(value) for name, value in zip(
                basis.BASIS, COEFFICIENTS, strict=True)},
        "coefficient_rule": (
            "exact integer rest-force null vector of the native C19 cube "
            "incidence action; no Dirac target used"),
        "rest_DEL_norm": float(np.linalg.norm(rest_del)),
        "block_identities": {
            "mass_symmetry_residual": float(np.linalg.norm(
                delta["mass"]-delta["mass"].T)),
            "gyro_antisymmetry_residual": float(np.linalg.norm(
                delta["gyro"]+delta["gyro"].T)),
            "stiffness_symmetry_residual": float(np.linalg.norm(
                delta["stiffness"]-delta["stiffness"].T)),
        },
        "baseline": {
            "common_connection_scale": natural,
            "maximum_growth_rate": float(
                np.log(max(abs(values)))/native_time.DT),
            "growing_multiplier_count": int(np.sum(growing)),
        },
        "projected_rows": projected,
        "full_spectrum_rows": full,
        "best_full_spectrum": best,
        "verdict": (
            "CUBE_INCIDENCE_SYMPLECTIC_GEOMETRIC_FULL_SPECTRUM_PASS"
            if any(row.get("growing_multiplier_count_at_1e_minus_7") == 0
                   and row["scale"] in candidates and row["scale"] != 0
                   for row in full)
            else "CUBE_INCIDENCE_SYMPLECTIC_GEOMETRIC_SCALES_RETAIN_GROWTH"),
        "claim_boundary": (
            "The integer internal combination is derived. The overall scale "
            "is screened only at native geometric values; a continuous "
            "inverse point has no provenance. A linear pass would still "
            "require nonlinear motion and current gates before admission."),
    }
    (HERE/"cube_incidence_symplectic_completion_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
