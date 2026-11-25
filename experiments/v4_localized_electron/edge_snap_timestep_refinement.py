"""Audit timestep scaling of the tensor relative-periodic residual."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def row(name: str):
    source = json.loads((HERE / name).read_text(encoding="utf-8"))
    dt = source["dt"]
    scaled = source["final_scaled_del_rms"]
    # The search divided DEL by B*chi0/dt.  A physical equation residual is
    # DEL/(dt*B*chi0), hence scaled/dt^2.
    return {
        "artifact": name,
        "updates": source["updates_per_translation"],
        "dt": dt,
        "translation_time": source["translation_time"],
        "stored_scaled_del_rms": scaled,
        "physical_equation_residual": scaled / dt**2,
        "charge_defects": source["maximum_normalized_noether_charge_defects"],
    }


def main() -> None:
    rows = [
        row("edge_snap_tensor_four_update_orbit.json"),
        row("edge_snap_tensor_eight_update_orbit.json"),
    ]
    ratio = (rows[1]["physical_equation_residual"]
             / rows[0]["physical_equation_residual"])
    result = {
        "status": "V4_EDGE_SNAP_TIMESTEP_REFINEMENT_FAIL",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "refined_over_coarse_physical_residual_ratio": ratio,
        "verdict": "RAW_DEL_SHRINKS_BUT_PHYSICAL_RESIDUAL_DOES_NOT_CONVERGE",
        "root_cause_update": (
            "The smaller eight-layer stored number is partly the automatic "
            "dt scaling of the discrete action residual. On the common "
            "physical equation scale the mismatch grows, so additional "
            "temporal subdivision cannot replace the missing chi response."),
        "next_action": (
            "Use the refined orbit residual as the inverse requirement for "
            "the next local reciprocal action basis; do not extend to sixteen "
            "updates until the physical residual decreases under refinement."),
    }
    (HERE / "edge_snap_timestep_refinement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
