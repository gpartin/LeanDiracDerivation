"""Check spatial-reflection covariance of the live-link action candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_box_transfer as native_box
import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_nonlinear_action as native_action
import live_clifford_bond_action as candidate


HERE = Path(__file__).resolve().parent
SIDE = 5
DT = native_time.DT


def pack(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def reflect(raw: torch.Tensor, axis: int) -> torch.Tensor:
    """Reflect sites and both native C3 vectors through the selected axis."""
    psi, chi = native_box.unpack(raw, SIDE)
    psi = torch.flip(psi, dims=(axis,))
    chi = torch.flip(chi, dims=(axis,))
    signs = torch.ones(3, dtype=torch.float64)
    signs[axis] = -1
    return pack(psi*signs, chi*signs)


def relative(left: float, right: float) -> float:
    return abs(left-right)/max(abs(left), abs(right), 1.0)


def main() -> None:
    rng = np.random.default_rng(4006041)
    dimension = SIDE**3*12
    rows = []
    for sample in range(3):
        left = torch.tensor(rng.normal(size=dimension), dtype=torch.float64)
        right = torch.tensor(rng.normal(size=dimension), dtype=torch.float64)
        for axis in range(3):
            original = float(candidate.interval(left, right, DT))
            transformed = float(candidate.interval(
                reflect(left, axis), reflect(right, axis), DT))
            rows.append({
                "sample": sample,
                "axis": axis,
                "interval_relative_defect": relative(original, transformed),
            })
    maximum = max(row["interval_relative_defect"] for row in rows)
    result = {
        "status": "V4_LIVE_LINK_REFLECTION_COVARIANCE_GATE",
        "link_source": candidate.LINK_SOURCE,
        "scale": candidate.SCALE,
        "maximum_interval_relative_defect": maximum,
        "rows": rows,
        "verdict": (
            "LIVE_LINK_REFLECTION_COVARIANT" if maximum < 1e-10
            else "LIVE_LINK_REFLECTION_COVARIANCE_FAIL"),
        "claim_boundary": (
            "Random-state action covariance under the three coordinate and "
            "native-vector reflections. It does not prove carrier motion."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    (HERE/f"live_link_reflection_covariance_gate_{suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
