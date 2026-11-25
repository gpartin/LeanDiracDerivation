"""Aggregate and red-team the two-resolution V4-003 readout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse_path = (args.coarse if args.coarse.is_absolute()
                   else HERE / args.coarse).resolve()
    fine_path = (args.fine if args.fine.is_absolute()
                 else HERE / args.fine).resolve()
    coarse, fine = load(coarse_path), load(fine_path)

    if coarse["dt"] <= fine["dt"]:
        raise ValueError("coarse timestep must exceed fine timestep")
    if abs(coarse["dt"] / fine["dt"] - 2.0) > 1e-12:
        raise ValueError("V4-003 gate requires a dt/dt-over-two pair")

    coarse_rows = {row["copy"]: row for row in coarse["rows"]}
    fine_rows = {row["copy"]: row for row in fine["rows"]}
    same_copies = set(coarse_rows) == set(fine_rows) == {"plus", "minus"}
    action_locked = (
        coarse["action_sha256"] == fine["action_sha256"]
        and coarse["rest_cache_sha256"] == fine["rest_cache_sha256"]
        and coarse["chi_scale_source"] == fine["chi_scale_source"]
        and coarse["phase_source"] == fine["phase_source"] == "native_graph"
    )
    rows = []
    for copy in ("plus", "minus"):
        first, second = coarse_rows[copy], fine_rows[copy]
        rows.append({
            "copy": copy,
            "four_complex_components": (
                first["four_complex_components"]
                and second["four_complex_components"]
                and first["real_component_count"] == 8
                and second["real_component_count"] == 8),
            "nonzero_live_chi": (
                abs(complex(first["live_chi_scale"]["real"],
                            first["live_chi_scale"]["imag"])) > 0
                and first["live_chi_scale"] == second["live_chi_scale"]),
            "coarse_exact_recovery_error": first["exact_recovery_error"],
            "fine_exact_recovery_error": second["exact_recovery_error"],
            "coarse_readout_error": first[
                "readout_evolution_relative_error"],
            "fine_readout_error": second[
                "readout_evolution_relative_error"],
            "readout_refinement_ratio": (
                second["readout_evolution_relative_error"]
                / first["readout_evolution_relative_error"]),
            "coarse_direct_phase_error": first[
                "direct_phase_relative_error"],
            "fine_direct_phase_error": second[
                "direct_phase_relative_error"],
            "direct_phase_refinement_ratio": (
                second["direct_phase_relative_error"]
                / first["direct_phase_relative_error"]),
            "coarse_complementary_residual": first[
                "complementary_sector_relative_residual"],
            "fine_complementary_residual": second[
                "complementary_sector_relative_residual"],
            "complementary_refinement_ratio": (
                second["complementary_sector_relative_residual"]
                / first["complementary_sector_relative_residual"]),
            "coarse_surrounding_fraction": first[
                "complementary_power_fractions"]["surroundings"],
            "fine_surrounding_fraction": second[
                "complementary_power_fractions"]["surroundings"],
            "readout_is_fixed_within_each_resolution": (
                len(first["fixed_readout_sha256"]) == 64
                and len(second["fixed_readout_sha256"]) == 64),
        })

    checks = {
        "same_two_cubical_copies": same_copies,
        "same_action_rest_and_chi_preparation": action_locked,
        "all_four_complex_components": all(
            row["four_complex_components"] for row in rows),
        "live_chi_is_in_the_fixed_embedding": all(
            row["nonzero_live_chi"] for row in rows),
        "lossless_left_inverse": all(
            max(row["coarse_exact_recovery_error"],
                row["fine_exact_recovery_error"]) < 1e-10
            for row in rows),
        "readout_error_is_second_order": all(
            row["readout_refinement_ratio"] < 0.30 for row in rows),
        "direct_phase_error_is_second_order": all(
            row["direct_phase_refinement_ratio"] < 0.30 for row in rows),
        "complementary_residual_shrinks": all(
            row["complementary_refinement_ratio"] < 0.60 for row in rows),
        "complement_is_measured_in_surroundings": all(
            row["coarse_surrounding_fraction"] > 0.5
            and row["fine_surrounding_fraction"] > 0.5
            for row in rows),
        "no_fitted_evolving_frame": (
            coarse["readout_construction"] == fine["readout_construction"]
            and "no trajectory fit" in coarse["readout_construction"]
            and coarse["chi_scale_source"]
                == "native_kinetic_normalization_1_over_sqrt_B"),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "stage": "V4-003",
        "status": status,
        "objective": (
            "One fixed localized two-layer Psi/chi map recovers four complex "
            "components for each cubical copy, while its explicitly retained "
            "complementary residual shrinks under timestep refinement."),
        "action_sha256": coarse["action_sha256"],
        "coarse_evidence": coarse_path.name,
        "fine_evidence": fine_path.name,
        "checks": checks,
        "rows": rows,
        "claim_boundary": (
            "V4-003 proves a fixed linear readout and controlled local "
            "remainder at the declared localized envelope. It does not yet "
            "prove operational spin, Noether-current pullback, or cell "
            "translation; those are V4-004 through V4-006."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
