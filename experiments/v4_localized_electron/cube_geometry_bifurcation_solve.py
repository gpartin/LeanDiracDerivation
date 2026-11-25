"""Solve the remaining transport instability on the cube-symmetric subspace.

The starting point is the best declared action screen: common Psi-density
connection, unit carrier/vacuum chi geometry, and the area-bilinear phase
square.  The full growing eigenspace is extracted without projection during
the original evolution.  Six scalar-oriented pair connections have rank five
on this space.  Their coefficients are solved only against native spectral
neutrality, then compared with the exact C19 cube eigenvalues 8 and 12.

This is an inverse/bifurcation diagnostic.  A numerical coefficient is not
action provenance; only a cube-geometric relation that passes the subsequent
full spectrum can survive the screen.
"""

from __future__ import annotations

import fractions
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigvalsh, lu_factor, lu_solve
from scipy.optimize import differential_evolution, minimize

import canonical_noncoplanar_bifurcation_audit as native_time
import complete_pair_wave_action_gate as spectrum
import current_action_localized_envelope_closure as envelope
import current_action_two_layer_spectrum as current
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
FEATURES = (
    "rho_Psi",
    "rho_chi",
    "area_density",
    "sigma_Psi_anisotropy",
    "sigma_chi_anisotropy",
    "area_bilinear_anisotropy",
)


def load_blocks(feature: str) -> dict[str, np.ndarray]:
    if feature == "rho_Psi":
        cache = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
        return {kind: cache[f"{kind}_common_pair_connection"]
                for kind in ("mass", "gyro", "stiffness")}
    if feature == "rho_chi":
        cache = np.load(HERE/"chi_density_pair_connection_matrices.npz")
    else:
        cache = np.load(HERE/f"scalar_pair_connection_{feature}_matrices.npz")
    return {kind: cache[kind] for kind in ("mass", "gyro", "stiffness")}


def base_matrices():
    bare = np.load(canonical.CACHE)
    activated = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    area_phase = np.load(HERE/"area_bilinear_phase_action_hessian.npz")[
        "normalized_hessian"]
    stiffness = (
        bare["stiffness"]
        + activated["hessian_carrier_carrier_chi"]
        + activated["hessian_carrier_carrier_Psi_tight_frame"]
        + activated["hessian_carrier_vacuum_chi"]
        + activated["hessian_vacuum_vacuum_chi"]
        + area_phase/native_time.CHI0**2)
    return bare, bare["mass"], bare["gyro"], stiffness


def invariant_hull(mass: np.ndarray, gyro: np.ndarray,
                   stiffness: np.ndarray, rest: np.ndarray):
    dressed = current.dressed_configuration(stiffness, mass, rest)
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(factor, (dt*stiffness-2*mass/dt)@dressed)
    backward = -lu_solve(factor, (mass/dt-gyro/2)@dressed)
    residual = np.column_stack((
        forward-dressed@(dressed.T@forward),
        backward-dressed@(dressed.T@backward)))
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    extra = u[:, singular > 1e-10*max(float(singular[0]), 1.0)]
    return envelope.orth(np.column_stack((dressed, extra)))


def monodromy(mass: np.ndarray, gyro: np.ndarray,
              stiffness: np.ndarray, q: np.ndarray):
    dt = native_time.DT
    mr, gr, kr = q.T@mass@q, q.T@gyro@q, q.T@stiffness@q
    factor = lu_factor(mr/dt+gr/2)
    forward = -lu_solve(factor, dt*kr-2*mr/dt)
    backward = -lu_solve(factor, mr/dt-gr/2)
    identity = np.eye(q.shape[1])
    return np.block([[forward, backward],
                     [identity, np.zeros_like(identity)]])


def real_growing_configuration(hull: np.ndarray,
                               reduced_monodromy: np.ndarray):
    values, vectors = eig(reduced_monodromy, check_finite=False)
    growing = abs(values) > 1+1e-7
    columns = []
    for vector in vectors[:, growing].T:
        present = hull@vector[:hull.shape[1]]
        previous = hull@vector[hull.shape[1]:]
        columns.extend((present.real, present.imag,
                        previous.real, previous.imag))
    q = envelope.orth(np.column_stack(columns))
    return values, growing, q


def reduced_blocks(blocks: dict[str, np.ndarray],
                   q: np.ndarray) -> dict[str, np.ndarray]:
    return {kind: q.T@blocks[kind]@q
            for kind in ("mass", "gyro", "stiffness")}


def reduced_monodromy(mass: np.ndarray, gyro: np.ndarray,
                      stiffness: np.ndarray):
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(factor, dt*stiffness-2*mass/dt)
    backward = -lu_solve(factor, mass/dt-gyro/2)
    identity = np.eye(mass.shape[0])
    return np.block([[forward, backward],
                     [identity, np.zeros_like(identity)]])


def spectral_objective(x: np.ndarray, base: dict[str, np.ndarray],
                       deltas: list[dict[str, np.ndarray]],
                       natural: float) -> float:
    mass = base["mass"].copy()
    gyro = base["gyro"].copy()
    stiffness = base["stiffness"].copy()
    for coefficient, blocks in zip(x, deltas, strict=True):
        mass += natural*coefficient*blocks["mass"]
        gyro += natural*coefficient*blocks["gyro"]
        stiffness += natural*coefficient*blocks["stiffness"]
    minimum_mass = float(eigvalsh(
        mass, subset_by_index=[0, 0], driver="evr")[0])
    if minimum_mass <= 1e-8:
        return 1000+abs(minimum_mass)*1000
    values = eig(reduced_monodromy(mass, gyro, stiffness),
                 right=False, check_finite=False)
    growth = float(np.log(max(abs(values)))/native_time.DT)
    # Prefer the smallest native correction among equally stable solutions.
    return max(growth, 0.0)+1e-6*float(np.dot(x, x))


def geometric_candidates(x: np.ndarray) -> list[dict]:
    candidates = []
    for denominator in (1, 2, 3, 4, 6, 8, 12, 19):
        rounded = np.round(x*denominator)/denominator
        candidates.append({
            "label": f"cube_rational_denominator_{denominator}",
            "coefficients": rounded.tolist(),
        })
    # Also record each independent best rational with denominator <= 19.
    candidates.append({
        "label": "independent_rational_denominator_at_most_19",
        "coefficients": [float(fractions.Fraction(float(value)).limit_denominator(19))
                         for value in x],
    })
    unique = []
    seen = set()
    for row in candidates:
        key = tuple(row["coefficients"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def full_row(coefficients: np.ndarray, label: str, bare,
             base: dict[str, np.ndarray], deltas, natural: float):
    mass = base["mass"].copy()
    gyro = base["gyro"].copy()
    stiffness = base["stiffness"].copy()
    for coefficient, blocks in zip(coefficients, deltas, strict=True):
        mass += natural*coefficient*blocks["mass"]
        gyro += natural*coefficient*blocks["gyro"]
        stiffness += natural*coefficient*blocks["stiffness"]
    minimum_mass = float(eigvalsh(
        mass, subset_by_index=[0, 0], driver="evr")[0])
    row = {
        "label": label,
        "coefficients_in_B_chi0_units": coefficients.tolist(),
        "minimum_full_mass_eigenvalue": minimum_mass,
    }
    if minimum_mass > 1e-9:
        dressed = current.dressed_configuration(stiffness, mass, bare["rest"])
        row.update(spectrum.spectrum_row(
            mass, gyro, stiffness, dressed, native_time.DT))
    return row


def main() -> None:
    bare, mass0, gyro0, stiffness0 = base_matrices()
    deltas = [load_blocks(feature) for feature in FEATURES]
    natural = native_time.B*native_time.CHI0
    initial = np.asarray((0.9, 0, 0, 0, 0, 0), dtype=float)
    base = {"mass": mass0, "gyro": gyro0, "stiffness": stiffness0}
    initial_mass = mass0+natural*initial[0]*deltas[0]["mass"]
    initial_gyro = gyro0+natural*initial[0]*deltas[0]["gyro"]
    initial_stiffness = stiffness0+natural*initial[0]*deltas[0]["stiffness"]
    hull = invariant_hull(
        initial_mass, initial_gyro, initial_stiffness, bare["rest"])
    initial_monodromy = monodromy(
        initial_mass, initial_gyro, initial_stiffness, hull)
    initial_values, growing, reduced = real_growing_configuration(
        hull, initial_monodromy)

    # Close the reduced configuration space once under all five independent
    # matrix responses.  This prevents an optimizer from exploiting a basis
    # direction omitted by the initial growing eigenvectors.
    columns = [reduced]
    for blocks in deltas:
        for kind in ("mass", "gyro", "stiffness"):
            columns.append(blocks[kind]@reduced)
    reduced = envelope.orth(np.column_stack(columns))

    reduced_base = reduced_blocks(base, reduced)
    reduced_deltas = [reduced_blocks(blocks, reduced) for blocks in deltas]
    objective = lambda x: spectral_objective(
        x, reduced_base, reduced_deltas, natural)
    bounds = [(0.0, 2.0)]+[(-2.0, 2.0)]*5
    global_fit = differential_evolution(
        objective, bounds=bounds, seed=4006, popsize=10,
        maxiter=45, tol=1e-7, polish=False, workers=1, updating="immediate")
    local_fit = minimize(
        objective, global_fit.x, method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-9, "fatol": 1e-10})
    solved = local_fit.x if local_fit.fun <= global_fit.fun else global_fit.x

    projected_rows = []
    candidate_rows = [{"label": "continuous_bifurcation_solution",
                       "coefficients": solved.tolist()}]
    candidate_rows.extend(geometric_candidates(solved))
    for candidate in candidate_rows:
        value = objective(np.asarray(candidate["coefficients"], dtype=float))
        projected_rows.append({**candidate, "projected_objective": float(value)})
    projected_rows.sort(key=lambda row: row["projected_objective"])

    # Full validation is expensive; test the continuous optimum and the four
    # best distinct cube/rational reductions.
    full_rows = [full_row(
        np.asarray(row["coefficients"], dtype=float), row["label"], bare,
        base, deltas, natural) for row in projected_rows[:5]]
    best_full = min(
        full_rows, key=lambda row: row.get("maximum_growth_rate", np.inf))

    response_columns = []
    for blocks in deltas:
        response_columns.append(np.concatenate(tuple(
            (reduced.T@blocks[kind]@reduced).ravel()
            for kind in ("mass", "gyro", "stiffness"))))
    response_singular = np.linalg.svd(
        np.column_stack(response_columns), compute_uv=False)
    result = {
        "status": "V4_CUBE_GEOMETRY_BIFURCATION_SOLVE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "features": list(FEATURES),
        "C19_cube_character_eigenvalues_used": [8, 12],
        "initial_growing_multiplier_count": int(np.sum(growing)),
        "initial_maximum_growth_rate": float(
            np.log(max(abs(initial_values)))/native_time.DT),
        "initial_growing_configuration_rank": int(reduced.shape[1]),
        "restricted_response_singular_values": response_singular.tolist(),
        "restricted_response_rank": int(np.sum(
            response_singular > 1e-10*max(float(response_singular[0]), 1.0))),
        "continuous_solution": {
            "coefficients_in_B_chi0_units": solved.tolist(),
            "projected_objective": float(objective(solved)),
            "optimizer_success": bool(local_fit.success or global_fit.success),
        },
        "projected_geometric_candidates": projected_rows,
        "full_spectrum_candidates": full_rows,
        "best_full_spectrum": best_full,
        "verdict": (
            "CUBE_GEOMETRY_BIFURCATION_FULL_SPECTRUM_PASS"
            if best_full.get("growing_multiplier_count_at_1e_minus_7") == 0
            else "CUBE_GEOMETRY_BIFURCATION_RETAINS_GROWTH"),
        "claim_boundary": (
            "The continuous solve is an inverse diagnostic, not coefficient "
            "provenance. Only the separately reported cube/rational candidates "
            "could qualify for a native action, and only after full refinements."),
    }
    (HERE/"cube_geometry_bifurcation_solve.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
