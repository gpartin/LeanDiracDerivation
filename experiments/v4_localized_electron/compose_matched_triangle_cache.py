"""Compose the native C19 triangle delta on the repaired frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREFIX = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0_"
    "overlap_0_overlapface_0_overlapmatched_1_overlappotential_1")
BASE = HERE / f"{PREFIX}_dt_0p00390625.npz"
TRIANGLE = HERE / "triangle_movement_blocks.npz"
TRIANGLE_BASE = HERE / (
    "wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
OUTPUT = HERE / f"{PREFIX}_triangle_1_dt_0p00390625.npz"


def main() -> None:
    names = ("previous", "current", "following")
    with np.load(BASE) as archive:
        base = {name: archive[name].copy() for name in names}
    with np.load(TRIANGLE) as archive:
        triangle = {name: archive[name].copy() for name in names}
    with np.load(TRIANGLE_BASE) as archive:
        triangle_base = {name: archive[name].copy() for name in names}
    blocks = {
        name: base[name]+triangle[name]-triangle_base[name]
        for name in names
    }
    np.savez_compressed(OUTPUT, **blocks)
    report = {
        "status": "V4_MATCHED_TRIANGLE_CACHE_COMPOSED",
        "identity": "candidate = repaired_frame + triangle - triangle_base",
        "output": OUTPUT.name,
    }
    (HERE / "compose_matched_triangle_cache.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
