"""Aggregate V4-005 short-time transport and native-current evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-coarse", type=Path, required=True)
    parser.add_argument("--transport-fine", type=Path, required=True)
    parser.add_argument("--current-coarse", type=Path, required=True)
    parser.add_argument("--current-fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path if path.is_absolute() else HERE/path for path in (
        args.transport_coarse, args.transport_fine,
        args.current_coarse, args.current_fine)]
    transport, transport_fine, current, current_fine = map(load, paths)
    stage2 = load(HERE/"v4_002_exact_1197/v4_002_stage_gate.json")
    stage3 = load(HERE/"v4_003_stage_gate.json")
    stage4 = load(HERE/"v4_004_stage_gate.json")

    state_ratio = (transport_fine["maximum_state_relative_error"]
                   / transport["maximum_state_relative_error"])
    readout_current_ratio = (
        transport_fine["maximum_readout_current_relative_error"]
        / transport["maximum_readout_current_relative_error"])
    density_ratio = (transport_fine["maximum_density_relative_error"]
                     / transport["maximum_density_relative_error"])
    center_ratio = (transport_fine["maximum_center_shift_absolute_error"]
                    / transport["maximum_center_shift_absolute_error"])
    noether_ratio = (
        current_fine["aggregate_cut_flux_relative_error"]
        / current["aggregate_cut_flux_relative_error"])
    action_hashes = {
        stage2["action_sha256"], stage3["action_sha256"],
        transport["action_sha256"], transport_fine["action_sha256"],
        current["action_sha256"], current_fine["action_sha256"]}
    checks = {
        "prior_chain_passes": (
            stage2["status"] == stage3["status"] == stage4["status"]
            == "PASS"),
        "same_frozen_action": len(action_hashes) == 1,
        "same_fixed_native_lift_at_both_dt": (
            transport["readouts"] == transport_fine["readouts"]),
        "both_copies_both_signs_three_axes": all(
            len(artifact["rows"]) == 12
            and {row["copy"] for row in artifact["rows"]}
            == {"plus", "minus"}
            and {row["axis"] for row in artifact["rows"]}
            == {"x", "y", "z"}
            and {row["sign"] for row in artifact["rows"]} == {-1, 1}
            for artifact in (transport, transport_fine, current, current_fine)),
        "short_time_state_matches": (
            transport["maximum_state_relative_error"] < 1e-4
            and transport_fine["maximum_state_relative_error"] < 3e-5
            and 0.20 < state_ratio < 0.30),
        "readout_density_and_current_match": (
            transport["maximum_readout_current_relative_error"] < 0.10
            and transport_fine[
                "maximum_readout_current_relative_error"] < 0.03
            and transport["maximum_density_relative_error"] < 1e-3
            and transport_fine["maximum_density_relative_error"] < 3e-4
            and all(0.20 < ratio < 0.30 for ratio in (
                readout_current_ratio, density_ratio, center_ratio))
            and transport["all_longitudinal_signs_match"]
            and transport_fine["all_longitudinal_signs_match"]),
        "action_derived_projected_noether_current_matches": (
            current["aggregate_cut_flux_relative_error"] < 0.01
            and current_fine["aggregate_cut_flux_relative_error"] < 0.005
            and noether_ratio < 0.60
            and current["all_cut_flux_signs_match"]
            and current_fine["all_cut_flux_signs_match"]),
        "projected_charge_remainder_shrinks": (
            current["maximum_native_charge_relative_change"] < 0.01
            and current_fine[
                "maximum_native_charge_relative_change"] < 0.005
            and current_fine["maximum_native_charge_relative_change"]
            < current["maximum_native_charge_relative_change"]),
        "Lean_theorem_declared": (
            HERE.parent.parent.joinpath(
                "LeanDirac/V4/LocalizedTransportCurrent.lean").is_file()),
    }
    passed = all(checks.values())
    result = {
        "stage": "V4-005",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "action_sha256": next(iter(action_hashes)),
        "metrics": {
            "coarse_state_error": transport["maximum_state_relative_error"],
            "fine_state_error": transport_fine[
                "maximum_state_relative_error"],
            "state_refinement_ratio": state_ratio,
            "coarse_readout_current_error": transport[
                "maximum_readout_current_relative_error"],
            "fine_readout_current_error": transport_fine[
                "maximum_readout_current_relative_error"],
            "readout_current_refinement_ratio": readout_current_ratio,
            "coarse_projected_noether_current_error": current[
                "aggregate_cut_flux_relative_error"],
            "fine_projected_noether_current_error": current_fine[
                "aggregate_cut_flux_relative_error"],
            "projected_noether_refinement_ratio": noether_ratio,
        },
        "Lean_theorem": (
            "LeanDirac.V4.localized_readout_transport_current"),
        "claim_boundary": (
            "Both cubical copies and both signs of all three axes agree with "
            "the independent exact C19/centered-time Dirac comparator for "
            "one local tangent update. The action-derived Noether current is "
            "pulled through the same center-resolved native lift. V4-005 does "
            "not prove a full cell crossing or nonlinear long-time motion."),
    }
    output = args.output if args.output.is_absolute() else HERE/args.output
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
