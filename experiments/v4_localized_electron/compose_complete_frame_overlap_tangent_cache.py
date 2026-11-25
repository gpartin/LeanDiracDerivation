"""Compose the exact tangent cache for the complete-frame overlap trial.

The interval action is linear in its action terms, so its second variation is
linear as well.  Reusing the separately audited overlap and incumbent blocks
avoids constructing one enormous nested-autodiff graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0_"
    "dt_0p00390625.npz")
OVERLAP = HERE / "overlap_material_tangent_blocks.npz"
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"
OUTPUT = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0_"
    "overlap_1_dt_0p00390625.npz")


def main() -> None:
    base = np.load(BASE)
    overlap = np.load(OVERLAP)
    incumbent = np.load(INCUMBENT)
    names = ("previous", "current", "following")
    blocks = {
        name: base[name] + overlap[name] - incumbent[name]
        for name in names
    }
    np.savez_compressed(OUTPUT, **blocks)
    report = {
        "status": "V4_COMPLETE_FRAME_OVERLAP_TANGENT_CACHE_COMPOSED",
        "identity": "combined = base + overlap_material - shared_incumbent",
        "base": BASE.name,
        "overlap": OVERLAP.name,
        "incumbent": INCUMBENT.name,
        "output": OUTPUT.name,
    }
    (HERE / "compose_complete_frame_overlap_tangent_cache.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
