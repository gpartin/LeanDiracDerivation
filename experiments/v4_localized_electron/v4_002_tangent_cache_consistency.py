"""Check a composed tangent cache against the declared nonlinear action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import live_clifford_bond_action as action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4002002)
    args = parser.parse_args()

    with np.load(args.rest_cache.resolve()) as archive:
        rest = archive["rest"].copy()
    with np.load(args.tangent_cache.resolve()) as archive:
        cached = {name: archive[name].copy()
                  for name in ("previous", "current", "following")}
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    rng = np.random.default_rng(args.seed)
    direction = rng.normal(size=rest.size)
    direction /= np.linalg.norm(direction)
    tangent = torch.tensor(direction, dtype=torch.float64)
    rows = []
    for selected, name in enumerate(("previous", "current", "following")):
        def function(value):
            trial = list(layers)
            trial[selected] = value
            return action.del_residual(*trial, dt)
        _, native_image = torch.func.jvp(
            function, (layers[selected],), (tangent,))
        native_image = native_image.detach().numpy()
        cached_image = cached[name]@direction
        rows.append({
            "block": name,
            "native_image_norm": float(np.linalg.norm(native_image)),
            "cached_image_norm": float(np.linalg.norm(cached_image)),
            "absolute_error": float(np.linalg.norm(
                native_image-cached_image)),
            "relative_error": float(np.linalg.norm(
                native_image-cached_image)
                / max(np.linalg.norm(native_image), 1e-300)),
        })
    maximum = max(row["relative_error"] for row in rows)
    result = {
        "stage": "V4-002",
        "status": "TANGENT_CACHE_CONSISTENCY",
        "action_configuration": {
            "link_source": action.LINK_SOURCE,
            "scale": action.SCALE,
            "spacetime_scale": action.SPACETIME_PLAQUETTE_SCALE,
            "parallel_matched_scale": action.PARALLEL_MATCHED_SCALE,
            "parallel_potential_scale": (
                action.PARALLEL_MATCHED_POTENTIAL_SCALE),
        },
        "rows": rows,
        "maximum_relative_error": maximum,
        "verdict": ("TANGENT_CACHE_MATCHES_ACTION"
                    if maximum < 1e-8 else
                    "TANGENT_CACHE_ACTION_MISMATCH"),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
