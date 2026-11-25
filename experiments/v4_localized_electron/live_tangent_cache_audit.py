"""Audit cached DEL tangent blocks against fresh directional derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as prior_action
import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent


def main() -> None:
    scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
    cache = HERE/f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}.npz"
    data = np.load(cache)
    blocks = [data[name] for name in ("previous", "current", "following")]
    rest = prior_action.expected_blocks()[0]["rest"]
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    arrays = (rotation.T@rest, rest, rotation@rest)
    tensors = tuple(torch.tensor(value, dtype=torch.float64) for value in arrays)
    rng = np.random.default_rng(4006006)
    rows = []
    for sample in range(3):
        vector = rng.normal(size=rest.size)
        vector /= np.linalg.norm(vector)
        direction = torch.tensor(vector, dtype=torch.float64)
        for selected in range(3):
            def function(value, index=selected):
                trial = list(tensors)
                trial[index] = value
                return action.del_residual(*trial, native_time.DT)
            _, fresh = torch.func.jvp(
                function, (tensors[selected],), (direction,))
            expected = blocks[selected]@vector
            rows.append({
                "sample": sample,
                "block": ("previous", "current", "following")[selected],
                "relative_error": float(
                    np.linalg.norm(fresh.detach().numpy()-expected)
                    / max(np.linalg.norm(expected), 1e-300)),
            })
    maximum = max(row["relative_error"] for row in rows)
    result = {
        "status": "V4_LIVE_TANGENT_CACHE_AUDIT",
        "link_source": action.LINK_SOURCE,
        "scale": action.SCALE,
        "cache": cache.name,
        "maximum_directional_relative_error": maximum,
        "rows": rows,
        "verdict": (
            "TANGENT_CACHE_CURRENT" if maximum < 1e-10
            else "TANGENT_CACHE_STALE_OR_INCORRECT"),
    }
    (HERE/"live_tangent_cache_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
