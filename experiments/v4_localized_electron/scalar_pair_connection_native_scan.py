"""Full spectra for native-scale independent scalar connection channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

import canonical_noncoplanar_bifurcation_audit as native_time
import complete_pair_wave_action_gate as spectrum
import current_action_two_layer_spectrum as current
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
FEATURES = (
    "rho_chi",
    "area_density",
    "sigma_Psi_anisotropy",
    "sigma_chi_anisotropy",
    "area_bilinear_anisotropy",
)


def load(feature: str) -> dict[str, np.ndarray]:
    if feature == "rho_chi":
        path = HERE/"chi_density_pair_connection_matrices.npz"
    else:
        path = HERE/f"scalar_pair_connection_{feature}_matrices.npz"
    cached = np.load(path)
    return {kind: cached[kind] for kind in ("mass", "gyro", "stiffness")}


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    psi_cache = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    psi = {kind: psi_cache[f"{kind}_common_pair_connection"]
           for kind in ("mass", "gyro", "stiffness")}
    natural = native_time.B*native_time.CHI0
    mass0 = bare["mass"]+natural*psi["mass"]
    gyro0 = bare["gyro"]+natural*psi["gyro"]
    stiffness0 = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"]
        + natural*psi["stiffness"])
    rows = []
    for feature in FEATURES:
        blocks = load(feature)
        for scale, provenance in (
                (native_time.CHI0,
                 "chi0 after chi-inertia normalization"),
                (natural, "B times chi0 native inertia-clock scale")):
            for sign in (-1.0, 1.0):
                coefficient = sign*scale
                mass = mass0+coefficient*blocks["mass"]
                gyro = gyro0+coefficient*blocks["gyro"]
                stiffness = stiffness0+coefficient*blocks["stiffness"]
                row = {
                    "feature": feature,
                    "coefficient": coefficient,
                    "coefficient_provenance": provenance,
                    "minimum_mass_eigenvalue": float(eigvalsh(
                        mass, subset_by_index=[0, 0], driver="evr")[0]),
                }
                if row["minimum_mass_eigenvalue"] > 1e-9:
                    dressed = current.dressed_configuration(
                        stiffness, mass, bare["rest"])
                    row.update(spectrum.spectrum_row(
                        mass, gyro, stiffness, dressed, native_time.DT))
                rows.append(row)
    best = min(rows, key=lambda row: row.get("maximum_growth_rate", np.inf))
    result = {
        "status": "V4_SCALAR_PAIR_CONNECTION_NATIVE_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "baseline_common_Psi_coefficient": natural,
        "rows": rows,
        "best": best,
        "verdict": (
            "INDEPENDENT_NATIVE_CONNECTION_FULL_SPECTRUM_PASS"
            if best.get("growing_multiplier_count_at_1e_minus_7") == 0
            else "INDEPENDENT_NATIVE_CONNECTIONS_RETAIN_GROWTH"),
        "claim_boundary": (
            "This is a declared native-scale sign screen, not coefficient "
            "fitting. No action is installed and nonlinear motion is open."),
    }
    (HERE/"scalar_pair_connection_native_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
