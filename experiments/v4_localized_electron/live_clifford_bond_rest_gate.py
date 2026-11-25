"""Rest-force gate for the live-Clifford bond action candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as native_action
import live_clifford_bond_action as candidate
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rest-cache", type=Path)
    parser.add_argument("--tangent-cache", type=Path)
    args = parser.parse_args()
    if args.rest_cache is not None:
        canonical.CACHE = args.rest_cache.resolve()
        with np.load(canonical.CACHE) as archive:
            rest = archive["rest"].copy()
    else:
        bare, _ = native_action.expected_blocks()
        rest = bare["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    residual = candidate.del_residual(*layers, dt).detach().numpy()
    backward_error = None
    if args.tangent_cache is not None:
        with np.load(args.tangent_cache.resolve()) as archive:
            blocks = tuple(archive[name] for name in (
                "previous", "current", "following"))
        scale = sum(np.linalg.norm(block, ord=np.inf)
                    * np.linalg.norm(layer.detach().numpy(), ord=np.inf)
                    for block, layer in zip(blocks, layers))
        backward_error = float(
            np.linalg.norm(residual, ord=np.inf)/max(scale, 1e-300))
    rest_null = (np.linalg.norm(residual)/np.sqrt(rest.size) < 1e-10
                 or (backward_error is not None
                     and backward_error < 1e-12))
    result = {
        "status": "V4_LIVE_CLIFFORD_BOND_REST_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "scale": candidate.SCALE,
        "chi_payload": candidate.CHI_PAYLOAD,
        "link_source": candidate.LINK_SOURCE,
        "rest_DEL_norm": float(np.linalg.norm(residual)),
        "rest_DEL_rms": float(np.linalg.norm(residual)/np.sqrt(rest.size)),
        "rest_DEL_backward_error": backward_error,
        "verdict": (
            "LIVE_CLIFFORD_BOND_REST_NULL"
            if rest_null else
            "LIVE_CLIFFORD_BOND_DEFORMS_REST"),
        "claim_boundary": (
            "Retained three-layer rest orbit checked by absolute residual "
            "and, for a supplied exact tangent cache, its scale-aware "
            "backward error. No movement or nonlinear endurance claim."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    output = (args.output if args.output is not None else
              HERE/f"live_clifford_bond_rest_gate_{suffix}.json")
    if not output.is_absolute():
        output = HERE/output
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
