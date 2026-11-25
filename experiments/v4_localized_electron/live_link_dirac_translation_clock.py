"""Derive the cell-crossing interval from the discrete C19 Dirac symbol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import localized_stability as c19


HERE = Path(__file__).resolve().parent


def main() -> None:
    momentum = np.asarray((0.35, 0.0, 0.0))
    symbol = dirac.c19_first_symbol(momentum)
    jacobian = sum(
        weight*np.outer(np.asarray(offset, dtype=float),
                        np.asarray(offset, dtype=float))
        * np.cos(momentum@np.asarray(offset, dtype=float))
        for offset, weight in c19.DIRECTIONS)/12.0
    energy = float(np.sqrt(native_time.CHI0**2+symbol@symbol))
    group_velocity = jacobian.T@symbol/energy
    speed = float(abs(group_velocity[0]))
    crossing_time = 1.0/speed
    maximum = {"momentum": 0.0, "speed": 0.0}
    for wave_number in np.linspace(0.0, np.pi, 10001):
        trial_momentum = np.asarray((wave_number, 0.0, 0.0))
        trial_symbol = dirac.c19_first_symbol(trial_momentum)
        trial_jacobian = sum(
            weight*np.outer(np.asarray(offset, dtype=float),
                            np.asarray(offset, dtype=float))
            * np.cos(trial_momentum@np.asarray(offset, dtype=float))
            for offset, weight in c19.DIRECTIONS)/12.0
        trial_energy = np.sqrt(native_time.CHI0**2+trial_symbol@trial_symbol)
        trial_speed = float(abs((trial_jacobian.T@trial_symbol/trial_energy)[0]))
        if trial_speed > maximum["speed"]:
            maximum = {"momentum": float(wave_number), "speed": trial_speed}
    result = {
        "status": "V4_DISCRETE_DIRAC_TRANSLATION_CLOCK",
        "carrier_momentum": momentum.tolist(),
        "C19_first_symbol": symbol.tolist(),
        "positive_energy": energy,
        "group_velocity": group_velocity.tolist(),
        "cell_crossing_time": crossing_time,
        "native_dt": native_time.DT,
        "updates_per_cell": crossing_time/native_time.DT,
        "speed_implied_by_one_cell_per_update": 1.0/native_time.DT,
        "maximum_axis_group_speed_grid_control": maximum,
        "minimum_cell_crossing_time_grid_control": 1.0/maximum["speed"],
        "minimum_updates_per_cell_grid_control": (
            1.0/maximum["speed"]/native_time.DT),
        "verdict": "CELL_CROSSING_INTERVAL_DERIVED_FROM_DISCRETE_DIRAC",
        "claim_boundary": (
            "Comparator timing only. The Dirac symbol is not inserted into "
            "the native action, and this does not prove a native handoff."),
    }
    (HERE/"live_link_dirac_translation_clock.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
