"""Verify the recorded V4-002 classification without converting FAIL to PASS."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def grouped(rows: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for row in rows:
        result.setdefault(row["side"], []).append(row)
    for values in result.values():
        values.sort(key=lambda row: row["dt"], reverse=True)
    return result


def main() -> None:
    audit = load("artifact_audit.json")
    for name, expected in audit["files"].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        require(actual == expected, f"artifact hash mismatch: {name}")
    canonical = load("canonical_stability.json")
    candidate = load("research_candidate_stability.json")
    multisite = load("multisite_response.json")
    kinetic = load("relative_kinetic_result.json")
    gyroscopic = load("gyroscopic_action_ledger.json")
    require(canonical["action_changed"] is False,
            "canonical artifact must use the unchanged action")
    require(candidate["action_changed"] is True,
            "research artifact must identify its changed action")
    require(len(canonical["rows"]) == len(candidate["rows"]) == 6,
            "matched artifacts require two boxes and three timesteps")
    require(multisite["status"] == "SCALAR_MULTISITE_STIFFNESS_NO_IMPROVEMENT",
            "multi-site response classification changed")
    require(multisite["positive_c19_curvature"]["best_growth_rate"]
            > multisite["baseline"]["leading_growth_rate"],
            "positive C19 curvature unexpectedly improved the spectrum")
    require(multisite["positive_cube_plaquette_curvature"]["best_growth_rate"]
            > multisite["baseline"]["leading_growth_rate"],
            "positive plaquette curvature unexpectedly improved the spectrum")
    require(multisite["inadmissible_inverse_sign_probe"]["relative_improvement"]
            < 0.01,
            "inverse-sign probe no longer supports the recorded conclusion")
    require(kinetic["validation"]["absolute_difference"] < 1e-6,
            "derived update no longer reproduces Verlet at zero coupling")
    require(kinetic["positive_probe"]["exact_rest_orbit_rms_residual"] < 1e-9,
            "relative-kinetic rest branch was not reproduced")
    require(kinetic["positive_probe"]["growth_ratio_to_baseline"] > 4.0,
            "relative-kinetic failure classification changed")
    require(gyroscopic["status"]
            == "GYROSCOPIC_ACTION_BASIS_PARTIAL_IMPROVEMENT",
            "gyroscopic action classification changed")
    require(gyroscopic["cycles"][-1]["relative_improvement"] > 0.06,
            "gyroscopic basis no longer reproduces the recorded improvement")
    require(gyroscopic["cycles"][-1]["growth_rate"] > 0.07,
            "gyroscopic diagnostic was incorrectly promoted to stable")

    for artifact in (canonical, candidate):
        for row in artifact["rows"]:
            require(row["exact_orbit_one_step_defect"] < 2e-9,
                    "exact rest orbit was not reproduced")
            require(row["mass_normalized_reciprocity_defect"] < 1e-12,
                    "reciprocal action tangent check failed")

    canonical_by_side = grouped(canonical["rows"])
    candidate_by_side = grouped(candidate["rows"])
    for side in (3, 4):
        base_rates = [row["leading_growth_rate_per_unit_time"]
                      for row in canonical_by_side[side]]
        trial_rates = [row["leading_growth_rate_per_unit_time"]
                       for row in candidate_by_side[side]]
        require(min(base_rates) > 1.0,
                "canonical finite growth rate was not reproduced")
        require(max(base_rates) - min(base_rates) < 0.01,
                "canonical rate does not converge in timestep")
        require(min(trial_rates) > 0.04,
                "candidate unexpectedly meets the spectral-stability gate")
        require(max(trial_rates) < min(base_rates) / 10,
                "candidate did not reproduce the recorded order-of-magnitude improvement")
        require(max(trial_rates) - min(trial_rates) < 0.001,
                "candidate rate does not converge in timestep")

    print("V4_STABILITY_CLASSIFICATION_REPRODUCED_CANONICAL_FAIL")


if __name__ == "__main__":
    main()
