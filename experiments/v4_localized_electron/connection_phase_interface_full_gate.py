"""Combine the derived transport, chi-interface, and Z2 phase responses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import complete_pair_wave_action_gate as spectrum
import current_action_two_layer_spectrum as current
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare = np.load(canonical.CACHE)
    activated = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    connection = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    common = {kind: connection[f"{kind}_common_pair_connection"]
              for kind in ("mass", "gyro", "stiffness")}
    tau = np.load(HERE/"relative_bilinear_phase_action_hessian.npz")[
        "normalized_hessian"]
    area = np.load(HERE/"area_bilinear_phase_action_hessian.npz")[
        "normalized_hessian"]
    natural = native_time.B*native_time.CHI0
    phase_scale = 1/native_time.CHI0**2
    phase_rows = (
        ("none", 0.0, 0.0),
        ("relative_bilinear", phase_scale, 0.0),
        ("area_bilinear", 0.0, phase_scale),
        ("both_bilinears", phase_scale, phase_scale),
    )
    rows = []
    for common_factor in (0.90, 0.95, 1.0):
        coefficient = common_factor*natural
        mass = bare["mass"]+coefficient*common["mass"]
        gyro = bare["gyro"]+coefficient*common["gyro"]
        base_stiffness = (
            bare["stiffness"]
            + activated["hessian_carrier_carrier_chi"]
            + activated["hessian_carrier_carrier_Psi_tight_frame"]
            + activated["hessian_carrier_vacuum_chi"]
            + activated["hessian_vacuum_vacuum_chi"]
            + coefficient*common["stiffness"])
        for label, tau_coefficient, area_coefficient in phase_rows:
            stiffness = (base_stiffness+tau_coefficient*tau
                         + area_coefficient*area)
            dressed = current.dressed_configuration(
                stiffness, mass, bare["rest"])
            row = spectrum.spectrum_row(
                mass, gyro, stiffness, dressed, native_time.DT)
            row.update({
                "common_factor": common_factor,
                "phase_label": label,
                "relative_bilinear_coefficient": tau_coefficient,
                "area_bilinear_coefficient": area_coefficient,
            })
            rows.append(row)
    best = min(rows, key=lambda row: row["maximum_growth_rate"])
    result = {
        "status": "V4_CONNECTION_PHASE_INTERFACE_FULL_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "phase_scale": phase_scale,
        "phase_scale_provenance": "1/chi0^2 geometric normalization",
        "rows": rows,
        "best": best,
        "verdict": (
            "DERIVED_CONNECTION_COMPLETION_FULL_SPECTRUM_PASS"
            if best["growing_multiplier_count_at_1e_minus_7"] == 0
            else "DERIVED_CONNECTION_COMPLETION_RETAINS_GROWTH"),
        "claim_boundary": (
            "Every screened term is a positive or symplectic reciprocal "
            "native action term. This remains a linear research screen and "
            "does not install an action or prove nonlinear motion."),
    }
    (HERE/"connection_phase_interface_full_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
