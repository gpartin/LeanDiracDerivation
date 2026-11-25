"""Identify which native bond information is missing from site transfer.

The incumbent contains the common density-oriented connection and the
geometric relative-phase control at factor chi0.  Each remaining scalar or
anisotropic native pair channel is added with both signs at the same B*chi0
scale.  The screen uses both four-component copies and literal Psi site
density.  It is a response classification, not an admitted action fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_geometry_bifurcation_solve as solve
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import discrete_site_transfer_pair_channel_screen as pair_screen


HERE = Path(__file__).resolve().parent
FEATURES = (
    "rho_chi",
    "area_density",
    "sigma_Psi_anisotropy",
    "sigma_chi_anisotropy",
    "area_bilinear_anisotropy",
)


def main() -> None:
    bare, _ = action.expected_blocks()
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    prepared = {}
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        prepared[flavor] = (raw, alpha, beta)

    natural = native_time.B * native_time.CHI0
    base = ablation.blocks()["base_plus_density_connection"]
    pair_cache = np.load(
        HERE / "density_oriented_pair_connection_matrices.npz")
    relative = tuple(pair_cache[f"{kind}_relative_pair_connection"]
                     for kind in ("mass", "gyro", "stiffness"))
    # chi0 is tested because the relative information is normalized once by
    # each of the chi0 local substrate modes.  This is a geometric diagnostic,
    # not an admitted coefficient derivation.
    relative_factor = native_time.CHI0
    incumbent = tuple(base[index] + natural * relative_factor
                      * relative[index] for index in range(3))
    dt = native_time.DT / 16
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))

    candidates = [("incumbent", 0.0, None)]
    candidates.extend((feature, sign, solve.load_blocks(feature))
                      for feature in FEATURES for sign in (-1.0, 1.0))
    rows = []
    for feature, sign, blocks in candidates:
        matrices = incumbent if blocks is None else tuple(
            incumbent[index] + natural * sign * blocks[kind]
            for index, kind in enumerate(("mass", "gyro", "stiffness")))
        cases = []
        for flavor, (raw, alpha, beta) in prepared.items():
            cases.append({
                "flavor": flavor,
                **pair_screen.evolve_case(
                    matrices, raw, alpha, beta, 0, 1, dt, steps),
            })
        rows.append({
            "feature": feature,
            "factor_in_B_chi0_units": sign,
            "finite_for_both_copies": all(
                case["finite_evolution"] for case in cases),
            "worst_copy_density_error": max(case[
                "maximum_C19_averaged_Psi_pixel_density_relative_error"]
                for case in cases),
            "worst_copy_center_error": max(case[
                "maximum_Psi_pixel_center_shift_error"] for case in cases),
            "cases": cases,
        })

    finite_rows = [row for row in rows if row["finite_for_both_copies"]]
    best = min(finite_rows, key=lambda row: row["worst_copy_density_error"])
    if best["feature"] == "incumbent":
        best_matrices = incumbent
    else:
        best_blocks = solve.load_blocks(best["feature"])
        best_matrices = tuple(
            incumbent[index] + natural * best["factor_in_B_chi0_units"]
            * best_blocks[kind]
            for index, kind in enumerate(("mass", "gyro", "stiffness")))

    held_rows = []
    for flavor, (raw, alpha, beta) in prepared.items():
        for axis, sign in ((0, -1), (0, 1), (3, 1)):
            held_rows.append({
                "flavor": flavor,
                **pair_screen.evolve_case(
                    best_matrices, raw, alpha, beta, axis, sign, dt, steps),
            })
    worst_held_density = max(row[
        "maximum_C19_averaged_Psi_pixel_density_relative_error"]
        for row in held_rows)
    worst_held_center = max(row["maximum_Psi_pixel_center_shift_error"]
                            for row in held_rows)
    baseline = next(row for row in rows if row["feature"] == "incumbent")
    result = {
        "status": "V4_DISCRETE_SITE_TRANSFER_INFORMATION_BASIS_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "incumbent": (
            "common density-oriented connection plus chi0 times the "
            "relative/conjugate connection; cube-incidence completion absent"),
        "candidate_scale": natural,
        "candidate_scale_provenance": "B times chi0",
        "rows": rows,
        "baseline_worst_copy_density_error": baseline[
            "worst_copy_density_error"],
        "best_feature": best["feature"],
        "best_feature_factor_in_B_chi0_units": best[
            "factor_in_B_chi0_units"],
        "best_training_worst_copy_density_error": best[
            "worst_copy_density_error"],
        "held_rows": held_rows,
        "worst_held_density_error": worst_held_density,
        "worst_held_center_error": worst_held_center,
        "verdict": (
            "NATIVE_INFORMATION_CHANNEL_SITE_TRANSFER_PASS"
            if worst_held_density < 0.05 and worst_held_center < 0.01
            else "NATIVE_INFORMATION_CHANNELS_DO_NOT_CLOSE_SITE_TRANSFER"),
        "claim_boundary": (
            "Unit-sign response screen around the chi0-relative incumbent. "
            "No coefficient is fitted or admitted. Both copies are training "
            "controls; held cases add reverse axial and diagonal motion for "
            "one linear cycle on side five."),
    }
    (HERE / "discrete_site_transfer_information_basis_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "held_rows")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
