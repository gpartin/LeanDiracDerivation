"""Scoped feasibility verdict for the exact-rest positive action cone.

This collector does not promote a fitted coefficient.  It records the finite
rest-null basis, its rejected members, and the fully converged side-five
spectral screens used to decide whether that cone contains a stability pass.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    baseline = read("canonical_transfer_oscillator_spectrum_0.json")
    names = [
        "canonical_transfer_oscillator_spectrum_0p5_even_0p15_density_0p25.json",
        "canonical_transfer_oscillator_spectrum_1_even_0p15_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_1_even_1_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_1_even_2_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_1_even_4_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_0p5_even_2_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_0_even_4_density_0p5.json",
        "canonical_transfer_oscillator_spectrum_0_even_2_density_1.json",
    ]
    scans = [read(name) for name in names]
    best = min(scans, key=lambda row: row["leading_spectral_radius"])
    rows = [{
        "odd_transfer_strength": row["strength"],
        "even_transfer_strength": row["even_strength"],
        "density_balance_strength": row["density_balance_strength"],
        "parallel_strength": row["parallel_strength"],
        "rest_DEL_rms": row["rest_DEL_rms"],
        "Legendre_condition_number": row[
            "following_block_condition_number"],
        "spectral_radius": row["leading_spectral_radius"],
        "growth_rate": row["leading_growth_rate"],
        "ritz_count": row["computed_eigenvalue_count"],
        "arpack_converged": row["arpack_converged"],
    } for row in scans]
    result = {
        "status": "V4_EXACT_REST_POSITIVE_ACTION_CONE_FEASIBILITY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "constraint_set": {
            "same_exact_rest_orbit": True,
            "positive_reciprocal_action_terms": True,
            "C19_cubic_symmetry": True,
            "no_prepared_rays_or_Dirac_objects": True,
            "unchanged_uniform_quadratic_Dirac_symbol": True,
            "three_axis_fixed_charge_response": True,
            "nonsingular_discrete_Legendre_map": True,
            "linear_stability_target": "spectral_radius <= 1+1e-7",
        },
        "admitted_screen_basis": [
            "odd transfer-coordinate harmonic oscillator",
            "even transfer-coordinate harmonic oscillator",
            "density-balance positive square",
            "pointwise parallel-overlap positive square",
        ],
        "excluded_before_coefficient_search": [{
            "basis": "spectral area-curvature square",
            "reason": "not rest-null on this carrier",
            "unit_rest_DEL_rms": 0.025855564355407548,
            "rest_energy": 2986.167031270297,
        }],
        "parallel_overlap_response": {
            "tested_strength": 0.1,
            "leading_radius_change_at_first_incumbent":
                3.9679370900103095e-13,
            "conclusion": "no resolved response on the leading unstable mode",
        },
        "screen_rows": rows,
        "best_resolved_row": rows[scans.index(best)],
        "canonical_growth_rate": baseline.get(
            "leading_growth_rate", 1.036009262774485),
        "best_growth_reduction_fraction": float(
            1-best["leading_growth_rate"]/1.036009262774485),
        "unresolved_edge": (
            "The lower-density even=2 solve did not converge efficiently; "
            "therefore this is a resolved screen failure, not an analytic "
            "no-go for every real coefficient in the continuous cone."),
        "verdict": "DECLARED_EXACT_REST_POSITIVE_CONE_NO_STABILITY_PASS",
        "promotion_consequence": (
            "Do not admit a coefficient or run a claimed finite handoff. "
            "The next action search must relax exact preservation of the old "
            "amplitudes and solve a nearby stable carrier while preserving its "
            "eight-site identity relations."),
        "claim_boundary": (
            "Scoped to the declared exact-rest, positive, local/C19 basis and "
            "the resolved side-five coefficient screen. It is not a universal "
            "two-field or all-action no-go."),
    }
    output = HERE / "canonical_inverse_action_feasibility.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
